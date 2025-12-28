"""Data export utilities for ground truth and email corpus."""

import json
import os
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from datetime import datetime

from src.models.email import Email, GroundTruth, EmailResponsiveness
from src.models.cpra import CPRARequest
from src.models.district import SchoolDistrict


class DataExporter:
    """Export generated data to various formats."""

    def __init__(self, output_dir: str = "data/generated"):
        """Initialize the data exporter."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_all(
        self,
        ground_truth: GroundTruth,
        district: SchoolDistrict,
        requests: List[CPRARequest]
    ) -> Dict[str, Path]:
        """Export all data in multiple formats."""
        exported_files = {}

        # Create timestamped subdirectory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_dir = self.output_dir / f"corpus_{timestamp}"
        export_dir.mkdir(parents=True, exist_ok=True)

        # Export emails as individual files
        emails_dir = export_dir / "emails"
        emails_dir.mkdir(exist_ok=True)
        self._export_emails_as_files(ground_truth.emails, emails_dir)
        exported_files["emails_dir"] = emails_dir

        # Export ground truth JSON
        gt_path = export_dir / "ground_truth.json"
        self._export_ground_truth_json(ground_truth, requests, gt_path)
        exported_files["ground_truth_json"] = gt_path

        # Export to Excel
        excel_path = export_dir / "email_corpus.xlsx"
        self._export_to_excel(ground_truth, district, requests, excel_path)
        exported_files["excel"] = excel_path

        # Export summary statistics
        stats_path = export_dir / "statistics.json"
        self._export_statistics(ground_truth, stats_path)
        exported_files["statistics"] = stats_path

        # Export CPRA requests
        requests_path = export_dir / "cpra_requests.json"
        self._export_cpra_requests(requests, requests_path)
        exported_files["cpra_requests"] = requests_path

        # Export district context
        district_path = export_dir / "district_context.json"
        self._export_district_context(district, district_path)
        exported_files["district_context"] = district_path

        return exported_files

    def _export_emails_as_files(self, emails: List[Email], output_dir: Path):
        """Export each email as an individual .eml-like text file."""
        for email in emails:
            filename = f"{email.id[:8]}.txt"
            filepath = output_dir / filename

            content = self._format_email_as_text(email)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)

    def _format_email_as_text(self, email: Email) -> str:
        """Format an email as a text file similar to .eml format."""
        lines = [
            f"Message-ID: {email.id}",
            f"Date: {email.date_sent.strftime('%a, %d %b %Y %H:%M:%S')}",
            f"From: {email.sender}",
            f"To: {', '.join(email.recipients)}",
        ]

        if email.cc:
            lines.append(f"Cc: {', '.join(email.cc)}")

        lines.extend([
            f"Subject: {email.subject}",
            f"X-Email-Type: {email.email_type.value}",
            f"X-Department: {email.department or 'None'}",
            f"X-Topics: {', '.join(email.topics)}",
            ""  # Blank line before body
        ])

        # Add body
        lines.append(email.body)

        # Add attachment info if present
        if email.attachments:
            lines.append("\n--- Attachments ---")
            for att in email.attachments:
                lines.append(f"  - {att.filename} ({att.file_type}, {att.size_kb} KB)")

        return "\n".join(lines)

    def _export_ground_truth_json(self, ground_truth: GroundTruth,
                                 requests: List[CPRARequest], output_path: Path):
        """Export ground truth to comprehensive JSON format."""
        data = {
            "metadata": {
                "generated_date": datetime.now().isoformat(),
                "total_emails": ground_truth.total_emails,
                "responsive_emails": ground_truth.responsive_emails,
                "responsive_rate": ground_truth.responsive_emails / ground_truth.total_emails if ground_truth.total_emails > 0 else 0,
                "total_requests": len(requests)
            },
            "emails": [],
            "cpra_requests": [],
            "responsiveness_map": {},
            "statistics": ground_truth.get_statistics()
        }

        # Add emails
        for email in ground_truth.emails:
            email_dict = {
                "id": email.id,
                "sender": email.sender,
                "recipients": email.recipients,
                "cc": email.cc,
                "subject": email.subject,
                "body": email.body,
                "date_sent": email.date_sent.isoformat(),
                "email_type": email.email_type.value,
                "department": email.department,
                "topics": email.topics,
                "generated_for_requests": email.generated_for_requests,
                "challenge_patterns": email.challenge_patterns,
                "has_attachments": email.has_attachments(),
                "attachments": [
                    {
                        "filename": att.filename,
                        "file_type": att.file_type,
                        "size_kb": att.size_kb
                    } for att in email.attachments
                ]
            }
            data["emails"].append(email_dict)

        # Add CPRA requests
        for request in requests:
            request_dict = {
                "id": request.id,
                "title": request.title,
                "description": request.description,
                "request_text": request.request_text,
                "date_submitted": request.date_submitted.isoformat(),
                "date_range_start": request.date_range_start.isoformat() if request.date_range_start else None,
                "date_range_end": request.date_range_end.isoformat() if request.date_range_end else None,
                "primary_keywords": request.primary_keywords,
                "secondary_keywords": request.secondary_keywords,
                "concepts": request.concepts,
                "complexity": request.complexity.value,
                "challenge_types": [ct.value for ct in request.challenge_types]
            }
            data["cpra_requests"].append(request_dict)

        # Add responsiveness map
        for email_id, responses in ground_truth.responsiveness_map.items():
            data["responsiveness_map"][email_id] = [
                {
                    "cpra_request_id": resp.cpra_request_id,
                    "is_responsive": resp.is_responsive,
                    "confidence": resp.confidence,
                    "reason": resp.reason.value,
                    "explanation": resp.explanation,
                    "matching_keywords": resp.matching_keywords,
                    "matching_concepts": resp.matching_concepts
                } for resp in responses
            ]

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _export_to_excel(self, ground_truth: GroundTruth, district: SchoolDistrict,
                        requests: List[CPRARequest], output_path: Path):
        """Export data to Excel with multiple sheets."""
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:

            # Sheet 1: Emails
            emails_data = []
            for email in ground_truth.emails:
                emails_data.append({
                    "Email ID": email.id[:8],
                    "Date": email.date_sent.strftime("%Y-%m-%d %H:%M"),
                    "Sender": email.sender,
                    "Recipients": ", ".join(email.recipients),
                    "Subject": email.subject,
                    "Body Preview": email.body[:100] + "..." if len(email.body) > 100 else email.body,
                    "Department": email.department or "N/A",
                    "Has Attachments": "Yes" if email.attachments else "No",
                    "Challenge Patterns": ", ".join(email.challenge_patterns) if email.challenge_patterns else "None"
                })

            df_emails = pd.DataFrame(emails_data)
            df_emails.to_excel(writer, sheet_name="Emails", index=False)

            # Sheet 2: Responsiveness Matrix
            matrix_data = []
            for email in ground_truth.emails:
                row = {"Email ID": email.id[:8]}
                for request in requests:
                    resp = ground_truth.get_responsiveness(email.id, request.id)
                    if resp and resp.is_responsive:
                        row[f"CPRA_{request.id[-3:]}"] = f"Yes ({resp.confidence:.2f})"
                    else:
                        row[f"CPRA_{request.id[-3:]}"] = "No"
                matrix_data.append(row)

            df_matrix = pd.DataFrame(matrix_data)
            df_matrix.to_excel(writer, sheet_name="Responsiveness Matrix", index=False)

            # Sheet 3: CPRA Requests
            requests_data = []
            for request in requests:
                requests_data.append({
                    "Request ID": request.id,
                    "Title": request.title,
                    "Date Range": f"{request.date_range_start.strftime('%Y-%m-%d') if request.date_range_start else 'N/A'} to "
                                 f"{request.date_range_end.strftime('%Y-%m-%d') if request.date_range_end else 'N/A'}",
                    "Primary Keywords": ", ".join(request.primary_keywords),
                    "Complexity": request.complexity.value,
                    "Challenge Types": ", ".join([ct.value for ct in request.challenge_types]),
                    "Responsive Emails": len(ground_truth.get_responsive_emails(request.id))
                })

            df_requests = pd.DataFrame(requests_data)
            df_requests.to_excel(writer, sheet_name="CPRA Requests", index=False)

            # Sheet 4: Statistics
            stats = ground_truth.get_statistics()
            stats_data = [
                {"Metric": "Total Emails", "Value": stats["total_emails"]},
                {"Metric": "Responsive Emails", "Value": stats["responsive_emails"]},
                {"Metric": "Response Rate", "Value": f"{stats['responsive_rate']:.2%}"},
            ]

            for request_id, count in stats["emails_per_request"].items():
                stats_data.append({
                    "Metric": f"Responsive to {request_id}",
                    "Value": count
                })

            df_stats = pd.DataFrame(stats_data)
            df_stats.to_excel(writer, sheet_name="Statistics", index=False)

            # Sheet 5: Staff Directory
            staff_data = []
            for staff in district.staff[:50]:  # Limit to first 50 for brevity
                staff_data.append({
                    "Name": staff.full_name,
                    "Email": staff.email,
                    "Role": staff.role.value,
                    "School": staff.school.name,
                    "Department": staff.department or "N/A"
                })

            df_staff = pd.DataFrame(staff_data)
            df_staff.to_excel(writer, sheet_name="Staff Directory", index=False)

    def _export_statistics(self, ground_truth: GroundTruth, output_path: Path):
        """Export detailed statistics."""
        stats = ground_truth.get_statistics()

        # Add more detailed statistics
        stats["generation_metadata"] = {
            "timestamp": datetime.now().isoformat(),
            "version": "1.0",
            "challenge_emails": sum(1 for e in ground_truth.emails if e.challenge_patterns),
            "emails_with_attachments": sum(1 for e in ground_truth.emails if e.attachments)
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)

    def _export_cpra_requests(self, requests: List[CPRARequest], output_path: Path):
        """Export CPRA requests to JSON."""
        requests_data = []
        for request in requests:
            requests_data.append({
                "id": request.id,
                "title": request.title,
                "description": request.description,
                "request_text": request.request_text,
                "date_submitted": request.date_submitted.isoformat(),
                "date_range_start": request.date_range_start.isoformat() if request.date_range_start else None,
                "date_range_end": request.date_range_end.isoformat() if request.date_range_end else None,
                "primary_keywords": request.primary_keywords,
                "secondary_keywords": request.secondary_keywords,
                "exclude_keywords": request.exclude_keywords,
                "concepts": request.concepts,
                "complexity": request.complexity.value,
                "challenge_types": [ct.value for ct in request.challenge_types],
                "department_targets": request.department_targets
            })

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(requests_data, f, indent=2)

    def _export_district_context(self, district: SchoolDistrict, output_path: Path):
        """Export district context to JSON."""
        district_data = {
            "name": district.name,
            "superintendent_email": district.superintendent_email,
            "schools": [
                {
                    "id": school.id,
                    "name": school.name,
                    "type": school.type.value,
                    "student_count": school.student_count,
                    "staff_count": school.staff_count
                } for school in district.schools
            ],
            "departments": [
                {
                    "name": dept.name,
                    "head_email": dept.head_email,
                    "member_count": len(dept.members)
                } for dept in district.departments
            ],
            "total_staff": len(district.staff),
            "topics": district.topics
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(district_data, f, indent=2)