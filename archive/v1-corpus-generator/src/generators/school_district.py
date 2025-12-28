"""Generator for school district context and personas."""

import random
from datetime import datetime, timedelta
from typing import List, Tuple
from faker import Faker

from src.models.district import (
    SchoolDistrict, School, StaffMember, Department,
    JobRole, SchoolType
)


class SchoolDistrictGenerator:
    """Generates a realistic school district context."""

    def __init__(self, seed: int = 42):
        """Initialize the generator with a seed for reproducibility."""
        self.fake = Faker()
        Faker.seed(seed)
        random.seed(seed)
        self.district_name = "Riverside Unified School District"
        self.domain = "rusd.edu"

    def generate_district(self) -> SchoolDistrict:
        """Generate a complete school district with schools and staff."""
        # Create schools first
        schools = self._generate_schools()

        # Create district office as a "school" entity
        district_office = School(
            id="district_office",
            name="District Office",
            type=SchoolType.DISTRICT_OFFICE,
            address="100 Education Way, Riverside, CA 92501",
            principal_email="",  # Will be set to superintendent
            phone="(951) 555-0100",
            grade_range="N/A",
            student_count=0,
            staff_count=25
        )
        schools.append(district_office)

        # Generate staff members
        staff = self._generate_staff(schools)

        # Set superintendent email
        superintendent = next(s for s in staff if s.role == JobRole.SUPERINTENDENT)
        district_office.principal_email = superintendent.email

        # Create departments
        departments = self._generate_departments(staff)

        # Create the district
        district = SchoolDistrict(
            name=self.district_name,
            superintendent_email=superintendent.email,
            schools=schools,
            staff=staff,
            departments=departments
        )

        return district

    def _generate_schools(self) -> List[School]:
        """Generate the schools in the district."""
        schools = []

        # Elementary Schools
        elementary_names = [
            "Riverside Elementary",
            "Oak Valley Elementary",
            "Sunset Ridge Elementary"
        ]
        for i, name in enumerate(elementary_names):
            schools.append(School(
                id=f"elementary_{i+1}",
                name=name,
                type=SchoolType.ELEMENTARY,
                address=f"{100 + i*10} School St, Riverside, CA 9250{i+1}",
                principal_email="",  # Will be set after staff generation
                phone=f"(951) 555-01{i+1}0",
                grade_range="K-5",
                student_count=random.randint(400, 600),
                staff_count=random.randint(30, 45)
            ))

        # Middle School
        schools.append(School(
            id="middle_1",
            name="Riverside Middle School",
            type=SchoolType.MIDDLE,
            address="200 Academy Ave, Riverside, CA 92504",
            principal_email="",  # Will be set after staff generation
            phone="(951) 555-0140",
            grade_range="6-8",
            student_count=random.randint(700, 900),
            staff_count=random.randint(45, 60)
        ))

        # High School
        schools.append(School(
            id="high_1",
            name="Riverside High School",
            type=SchoolType.HIGH,
            address="300 Titan Way, Riverside, CA 92505",
            principal_email="",  # Will be set after staff generation
            phone="(951) 555-0150",
            grade_range="9-12",
            student_count=random.randint(1200, 1500),
            staff_count=random.randint(70, 90)
        ))

        return schools

    def _generate_staff(self, schools: List[School]) -> List[StaffMember]:
        """Generate staff members for all schools."""
        staff = []
        used_emails = set()

        # District Office Leadership
        district_office = next(s for s in schools if s.type == SchoolType.DISTRICT_OFFICE)

        # Superintendent
        staff.append(self._create_staff_member(
            JobRole.SUPERINTENDENT, district_office, used_emails,
            first_name="Dr. Patricia", last_name="Johnson"
        ))

        # Assistant Superintendents
        for i, area in enumerate(["Curriculum", "Business Services"]):
            staff.append(self._create_staff_member(
                JobRole.ASSISTANT_SUPERINTENDENT, district_office, used_emails,
                department=area
            ))

        # District Directors
        director_roles = [
            (JobRole.IT_DIRECTOR, "Technology"),
            (JobRole.FACILITIES_MANAGER, "Facilities"),
            (JobRole.FINANCE_DIRECTOR, "Finance"),
            (JobRole.HR_DIRECTOR, "Human Resources"),
            (JobRole.CURRICULUM_DIRECTOR, "Curriculum"),
            (JobRole.SAFETY_COORDINATOR, "Safety"),
            (JobRole.TRANSPORTATION_DIRECTOR, "Transportation")
        ]
        for role, dept in director_roles:
            staff.append(self._create_staff_member(
                role, district_office, used_emails, department=dept
            ))

        # Generate staff for each school
        for school in schools:
            if school.type == SchoolType.DISTRICT_OFFICE:
                continue

            # Principal and Vice Principal
            principal = self._create_staff_member(
                JobRole.PRINCIPAL, school, used_emails
            )
            staff.append(principal)
            school.principal_email = principal.email

            staff.append(self._create_staff_member(
                JobRole.VICE_PRINCIPAL, school, used_emails,
                supervisor_email=principal.email
            ))

            # Teachers (based on school type)
            if school.type == SchoolType.ELEMENTARY:
                num_teachers = random.randint(20, 30)
            elif school.type == SchoolType.MIDDLE:
                num_teachers = random.randint(30, 40)
            else:  # HIGH
                num_teachers = random.randint(50, 65)

            for _ in range(num_teachers):
                staff.append(self._create_staff_member(
                    JobRole.TEACHER, school, used_emails,
                    supervisor_email=principal.email
                ))

            # Special Education Teachers
            for _ in range(random.randint(2, 5)):
                staff.append(self._create_staff_member(
                    JobRole.SPECIAL_ED_TEACHER, school, used_emails,
                    department="Special Education",
                    supervisor_email=principal.email
                ))

            # Support Staff
            support_roles = [
                JobRole.COUNSELOR,
                JobRole.LIBRARIAN,
                JobRole.NURSE,
                JobRole.SECRETARY,
                JobRole.IT_SUPPORT,
                JobRole.CAFETERIA_MANAGER
            ]
            for role in support_roles:
                staff.append(self._create_staff_member(
                    role, school, used_emails,
                    supervisor_email=principal.email
                ))

            # Maintenance Staff (1-2 per school)
            for _ in range(random.randint(1, 2)):
                staff.append(self._create_staff_member(
                    JobRole.MAINTENANCE_STAFF, school, used_emails,
                    department="Facilities"
                ))

        return staff

    def _create_staff_member(
        self,
        role: JobRole,
        school: School,
        used_emails: set,
        first_name: str = None,
        last_name: str = None,
        department: str = None,
        supervisor_email: str = None
    ) -> StaffMember:
        """Create a single staff member."""
        if not first_name:
            first_name = self.fake.first_name()
        if not last_name:
            last_name = self.fake.last_name()

        # Generate unique email
        base_email = f"{first_name.lower()}.{last_name.lower()}@{self.domain}"
        email = base_email
        counter = 1
        while email in used_emails:
            email = f"{first_name.lower()}.{last_name.lower()}{counter}@{self.domain}"
            counter += 1
        used_emails.add(email)

        # Generate hire date (within last 15 years)
        days_ago = random.randint(30, 15 * 365)
        start_date = datetime.now() - timedelta(days=days_ago)

        return StaffMember(
            id=f"staff_{len(used_emails)}",
            first_name=first_name,
            last_name=last_name,
            email=email,
            role=role,
            school=school,
            department=department,
            phone=self.fake.phone_number(),
            start_date=start_date,
            supervisor_email=supervisor_email
        )

    def _generate_departments(self, staff: List[StaffMember]) -> List[Department]:
        """Generate departments based on staff."""
        departments = []

        dept_definitions = [
            ("Technology", JobRole.IT_DIRECTOR),
            ("Facilities", JobRole.FACILITIES_MANAGER),
            ("Finance", JobRole.FINANCE_DIRECTOR),
            ("Human Resources", JobRole.HR_DIRECTOR),
            ("Curriculum", JobRole.CURRICULUM_DIRECTOR),
            ("Special Education", JobRole.SPECIAL_ED_TEACHER),
            ("Transportation", JobRole.TRANSPORTATION_DIRECTOR),
            ("Safety", JobRole.SAFETY_COORDINATOR)
        ]

        for dept_name, lead_role in dept_definitions:
            # Find department head
            head = next((s for s in staff if s.role == lead_role), None)
            if not head:
                continue

            # Find department members
            members = [s.email for s in staff
                      if s.department == dept_name or s.role == lead_role]

            departments.append(Department(
                name=dept_name,
                head_email=head.email,
                members=members,
                budget_code=f"DEPT-{dept_name[:3].upper()}",
                description=f"Manages {dept_name.lower()} for the district"
            ))

        return departments