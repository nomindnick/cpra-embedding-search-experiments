"""Data models for school district context."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
from enum import Enum
from datetime import datetime


class JobRole(Enum):
    """Enumeration of job roles in the school district."""
    # Administration
    SUPERINTENDENT = "Superintendent"
    ASSISTANT_SUPERINTENDENT = "Assistant Superintendent"
    PRINCIPAL = "Principal"
    VICE_PRINCIPAL = "Vice Principal"
    DISTRICT_ADMIN = "District Administrator"

    # Teaching Staff
    TEACHER = "Teacher"
    SUBSTITUTE_TEACHER = "Substitute Teacher"
    SPECIAL_ED_TEACHER = "Special Education Teacher"
    COUNSELOR = "Counselor"
    LIBRARIAN = "Librarian"

    # Support Staff
    IT_DIRECTOR = "IT Director"
    IT_SUPPORT = "IT Support Specialist"
    FACILITIES_MANAGER = "Facilities Manager"
    MAINTENANCE_STAFF = "Maintenance Staff"
    CAFETERIA_MANAGER = "Cafeteria Manager"
    NURSE = "School Nurse"
    SECRETARY = "Secretary"

    # Specialized Roles
    CURRICULUM_DIRECTOR = "Curriculum Director"
    FINANCE_DIRECTOR = "Finance Director"
    HR_DIRECTOR = "Human Resources Director"
    SAFETY_COORDINATOR = "Safety Coordinator"
    TRANSPORTATION_DIRECTOR = "Transportation Director"


class SchoolType(Enum):
    """Types of schools in the district."""
    ELEMENTARY = "Elementary School"
    MIDDLE = "Middle School"
    HIGH = "High School"
    DISTRICT_OFFICE = "District Office"


@dataclass
class School:
    """Represents a school in the district."""
    id: str
    name: str
    type: SchoolType
    address: str
    principal_email: str
    phone: str
    grade_range: str
    student_count: int
    staff_count: int

    def __hash__(self):
        return hash(self.id)


@dataclass
class StaffMember:
    """Represents a staff member in the school district."""
    id: str
    first_name: str
    last_name: str
    email: str
    role: JobRole
    school: School
    department: Optional[str] = None
    phone: Optional[str] = None
    start_date: Optional[datetime] = None
    supervisor_email: Optional[str] = None

    @property
    def full_name(self) -> str:
        """Get the full name of the staff member."""
        return f"{self.first_name} {self.last_name}"

    def __hash__(self):
        return hash(self.email)


@dataclass
class Department:
    """Represents a department in the district."""
    name: str
    head_email: str
    members: List[str] = field(default_factory=list)  # List of email addresses
    budget_code: Optional[str] = None
    description: Optional[str] = None


@dataclass
class SchoolDistrict:
    """Represents the entire school district."""
    name: str
    superintendent_email: str
    schools: List[School] = field(default_factory=list)
    staff: List[StaffMember] = field(default_factory=list)
    departments: List[Department] = field(default_factory=list)
    fiscal_year_start: datetime = field(default_factory=lambda: datetime(2024, 7, 1))
    fiscal_year_end: datetime = field(default_factory=lambda: datetime(2025, 6, 30))

    # Common district-wide topics for email generation
    topics: List[str] = field(default_factory=lambda: [
        "budget planning",
        "student achievement",
        "facility maintenance",
        "technology infrastructure",
        "professional development",
        "safety protocols",
        "special education",
        "curriculum updates",
        "parent communication",
        "standardized testing",
        "food services",
        "transportation",
        "substitute coverage",
        "grant applications",
        "vendor contracts"
    ])

    def get_staff_by_role(self, role: JobRole) -> List[StaffMember]:
        """Get all staff members with a specific role."""
        return [s for s in self.staff if s.role == role]

    def get_staff_by_school(self, school: School) -> List[StaffMember]:
        """Get all staff members at a specific school."""
        return [s for s in self.staff if s.school == school]

    def get_staff_by_email(self, email: str) -> Optional[StaffMember]:
        """Get a staff member by email address."""
        for staff in self.staff:
            if staff.email == email:
                return staff
        return None

    def get_department_members(self, department_name: str) -> List[StaffMember]:
        """Get all members of a specific department."""
        dept = next((d for d in self.departments if d.name == department_name), None)
        if dept:
            return [self.get_staff_by_email(email) for email in dept.members
                    if self.get_staff_by_email(email)]
        return []