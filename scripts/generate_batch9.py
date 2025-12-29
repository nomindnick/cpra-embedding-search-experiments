#!/usr/bin/env python3
"""Generate Batch 9: TRUE_NEGATIVE emails for the CPRA corpus."""

import json

# TRUE_NEGATIVE emails - clearly unrelated to water, lead, or contamination
# These should be baseline negatives that neither keyword nor embedding search should match

new_emails = [
    # HR/Personnel Emails (285-294)
    {
        "id": "email_285",
        "thread_id": None,
        "thread_position": None,
        "thread_length": None,
        "date": "2023-01-09T09:00:00",
        "from": "hr@cityofexample.gov",
        "to": ["all-staff@cityofexample.gov"],
        "cc": [],
        "subject": "Open Enrollment Reminder - Benefits Selection Due Jan 31",
        "body": """All Staff,

This is a reminder that open enrollment for 2023 benefits closes January 31st.

What you need to do:
- Log into the benefits portal at benefits.cityofexample.gov
- Review your current elections
- Make any changes for the new plan year
- Submit your selections before the deadline

Key changes this year:
- New dental provider (Delta Dental replacing MetLife)
- HSA contribution limits increased to $3,850 individual / $7,750 family
- Vision coverage now includes one pair of sunglasses per year
- EAP expanded to 8 free counseling sessions (up from 5)

If you take no action, your current elections will roll over EXCEPT for FSA contributions, which must be re-elected annually.

Benefits fair: January 18th in the Community Room, 11am-2pm
Representatives from all carriers will be present to answer questions.

Contact HR with any questions.

Human Resources Department""",
        "has_attachment": True,
        "attachment_names": ["2023_Benefits_Guide.pdf"]
    },
    {
        "id": "email_286",
        "thread_id": None,
        "thread_position": None,
        "thread_length": None,
        "date": "2023-02-15T14:30:00",
        "from": "hr@cityofexample.gov",
        "to": ["department-heads@cityofexample.gov"],
        "cc": [],
        "subject": "Performance Review Cycle - Manager Training Required",
        "body": """Department Heads,

The annual performance review cycle begins March 1st. Before then, all supervisors must complete the updated performance management training.

Training details:
- Online module: 2 hours (available now in LMS)
- Deadline to complete: February 28th
- Topics: new rating scale, goal-setting, feedback conversations, documentation

Key changes this cycle:
- Moving from 5-point to 4-point rating scale
- Eliminating "meets expectations" - now "successful" or "developing"
- Adding mid-year check-in requirement
- New self-assessment form for employees

Timeline:
- March 1-31: Employee self-assessments due
- April 1-30: Manager assessments and calibration
- May 1-15: Review meetings with employees
- May 31: All reviews finalized in system

Calibration sessions will be scheduled by department. I'll send calendar invites next week.

Please ensure all supervisors in your department complete the training on time.

Sandra Mitchell
HR Director""",
        "has_attachment": False,
        "attachment_names": []
    },
    {
        "id": "email_287",
        "thread_id": None,
        "thread_position": None,
        "thread_length": None,
        "date": "2023-03-01T10:15:00",
        "from": "hr@cityofexample.gov",
        "to": ["all-staff@cityofexample.gov"],
        "cc": [],
        "subject": "Welcome New Employees - March 2023",
        "body": """Please join me in welcoming our newest team members:

CITY MANAGER'S OFFICE
- Jessica Park, Executive Assistant (started 2/27)

FINANCE DEPARTMENT
- Michael Torres, Senior Accountant (starts 3/6)
- Aisha Patel, Budget Analyst (starts 3/13)

POLICE DEPARTMENT
- Officers Ryan Chen, Maria Gonzalez, and David Williams (Academy Class 23-1, graduated 2/24)

PARKS & RECREATION
- Thomas Anderson, Recreation Coordinator (started 2/27)
- Lisa Kim, Groundskeeper (starts 3/6)

PLANNING DEPARTMENT
- Robert Martinez, Associate Planner (starts 3/13)

PUBLIC WORKS
- James O'Brien, Equipment Operator (started 2/27)

New employee orientation is held the first Monday of each month. Please help our new colleagues feel welcome and show them around.

If you see someone new, introduce yourself!

Human Resources""",
        "has_attachment": False,
        "attachment_names": []
    },
    {
        "id": "email_288",
        "thread_id": None,
        "thread_position": None,
        "thread_length": None,
        "date": "2023-04-28T16:00:00",
        "from": "hr@cityofexample.gov",
        "to": ["all-staff@cityofexample.gov"],
        "cc": [],
        "subject": "Retirement Celebration - Chief Williams",
        "body": """All Staff,

Please join us in celebrating Police Chief Margaret Williams' retirement after 32 years of dedicated service to our community.

RETIREMENT RECEPTION
Date: Friday, May 12th
Time: 3:00 - 5:00 PM
Location: City Hall Atrium

Chief Williams joined the department in 1991 as a patrol officer and rose through the ranks to become our first female Police Chief in 2015. Under her leadership, the department implemented community policing initiatives, reduced response times by 23%, and earned state accreditation.

A memory book is being compiled. Please email your stories, photos, or well-wishes to hr@cityofexample.gov by May 5th.

A gift collection is underway - see your department admin if you'd like to contribute.

Light refreshments will be served. RSVP appreciated but not required.

Thank you, Chief Williams, for your service!

Human Resources""",
        "has_attachment": False,
        "attachment_names": []
    },
    {
        "id": "email_289",
        "thread_id": None,
        "thread_position": None,
        "thread_length": None,
        "date": "2023-05-15T08:30:00",
        "from": "training@cityofexample.gov",
        "to": ["all-staff@cityofexample.gov"],
        "cc": [],
        "subject": "Mandatory Sexual Harassment Prevention Training Due June 30",
        "body": """All City Employees,

Per state law AB 1825 and SB 1343, all employees must complete sexual harassment prevention training every two years.

Requirements:
- Supervisors: 2-hour course
- Non-supervisory employees: 1-hour course

How to complete:
1. Log into the Learning Management System (LMS)
2. Find the course under "Required Training"
3. Complete all modules and final quiz
4. Print your certificate of completion

Deadline: June 30, 2023

The course covers:
- Definition and examples of harassment
- Bystander intervention strategies
- Reporting procedures
- Manager responsibilities
- Retaliation protections

Completion status is tracked automatically. Department heads will receive weekly reports of outstanding completions starting June 1st.

Those who completed training in 2022 are not required to retake it this year.

Questions? Contact training@cityofexample.gov

Training & Development""",
        "has_attachment": False,
        "attachment_names": []
    },
    {
        "id": "email_290",
        "thread_id": None,
        "thread_position": None,
        "thread_length": None,
        "date": "2023-06-21T11:45:00",
        "from": "hr@cityofexample.gov",
        "to": ["all-staff@cityofexample.gov"],
        "cc": [],
        "subject": "Holiday Schedule - Independence Day",
        "body": """All Staff,

City offices will be closed Tuesday, July 4th in observance of Independence Day.

Essential services will operate as follows:
- Police/Fire: Normal staffing (holiday pay applies)
- Transit: Sunday schedule
- Parks: Facilities closed, trails remain open
- Solid Waste: Tuesday routes collected Wednesday; all others normal

For those working the holiday:
- Holiday premium pay per MOU provisions
- Comp time in lieu of premium available upon request

The City's fireworks celebration is Tuesday evening at Memorial Park. Employees and families are welcome - look for the employee section near the main stage.

Stay safe and enjoy the holiday!

Human Resources""",
        "has_attachment": False,
        "attachment_names": []
    },
    {
        "id": "email_291",
        "thread_id": None,
        "thread_position": None,
        "thread_length": None,
        "date": "2023-07-10T09:00:00",
        "from": "hr@cityofexample.gov",
        "to": ["supervisors@cityofexample.gov"],
        "cc": [],
        "subject": "New Telework Policy Effective August 1",
        "body": """Supervisors,

Following the Council's adoption of the updated Administrative Policy 3.15, our revised telework policy takes effect August 1st.

Key provisions:
- Eligible positions may telework up to 2 days per week
- Supervisor approval required; not an entitlement
- Core hours (10am-3pm) must be accessible for meetings
- City-issued equipment required for remote work
- Performance expectations unchanged

Eligibility criteria:
- Completed probationary period
- Satisfactory performance rating
- Job duties compatible with remote work
- Reliable internet and appropriate workspace

Positions NOT eligible:
- Public-facing customer service
- Field operations requiring physical presence
- Positions requiring access to confidential records

Request process:
1. Employee submits Telework Agreement form
2. Supervisor reviews and approves/denies within 10 days
3. IT coordinates equipment if needed
4. Trial period of 90 days

Forms available on the intranet. Training webinar scheduled for July 24th at 2pm.

HR Director""",
        "has_attachment": True,
        "attachment_names": ["Telework_Agreement_Form.pdf", "Policy_3.15_Telework.pdf"]
    },
    {
        "id": "email_292",
        "thread_id": None,
        "thread_position": None,
        "thread_length": None,
        "date": "2023-08-08T14:20:00",
        "from": "benefits@cityofexample.gov",
        "to": ["all-staff@cityofexample.gov"],
        "cc": [],
        "subject": "New Employee Assistance Program Provider",
        "body": """All Employees,

Effective September 1st, our Employee Assistance Program (EAP) provider will change from LifeWorks to ComPsych.

What's changing:
- New phone number: 1-888-555-4567
- New website: guidanceresources.com
- More counseling sessions: 8 per issue (up from 5)
- Additional services (see below)

What's staying the same:
- Completely confidential
- Available to employees and household members
- No cost to you
- 24/7 availability

New services included:
- Financial counseling and debt management
- Legal consultations (30 min free)
- Identity theft recovery assistance
- Caregiver support resources
- Relationship coaching

LifeWorks access ends August 31st. If you're currently in counseling, your counselor can help with transition planning.

Welcome webinars:
- September 5th at noon
- September 7th at 5pm

Your well-being matters. These resources are here to help.

Benefits Team""",
        "has_attachment": False,
        "attachment_names": []
    },
    {
        "id": "email_293",
        "thread_id": None,
        "thread_position": None,
        "thread_length": None,
        "date": "2023-09-05T10:00:00",
        "from": "hr@cityofexample.gov",
        "to": ["union-members@cityofexample.gov"],
        "cc": ["union-rep@seiu721.org"],
        "subject": "MOU Ratification Results",
        "body": """SEIU Local 721 Members,

The tentative agreement has been ratified by a vote of 234-56.

The new Memorandum of Understanding is effective October 1, 2023 through September 30, 2026.

Summary of key provisions:
- Wages: 4% increase Year 1, 3% Year 2, 3% Year 3
- Health benefits: City contribution increases to cover 90% of Kaiser rate
- Vacation accrual: Increased at 15-year mark
- Shift differential: Night premium increased to $1.50/hr
- Education reimbursement: Increased to $3,000/year

Implementation timeline:
- October 1: New MOU effective
- October 15: Wage increase reflected in paycheck
- November 1: Benefits changes effective
- Retroactive pay (if any): December paycheck

Thank you to the bargaining team members who worked diligently on negotiations over the past four months.

Sandra Mitchell
HR Director""",
        "has_attachment": True,
        "attachment_names": ["SEIU_MOU_2023-2026.pdf"]
    },
    {
        "id": "email_294",
        "thread_id": None,
        "thread_position": None,
        "thread_length": None,
        "date": "2023-10-02T08:45:00",
        "from": "wellness@cityofexample.gov",
        "to": ["all-staff@cityofexample.gov"],
        "cc": [],
        "subject": "Wellness Program - Fall Activities",
        "body": """City Staff,

Our wellness program has exciting activities this fall:

STEP CHALLENGE - October 15-November 15
- Team-based competition (5 people per team)
- Track steps using any device
- Prizes for top 3 teams
- Register by October 10th

FLU SHOT CLINIC - October 25th
- City Hall Conference Room A, 9am-3pm
- Free for employees and covered dependents
- No appointment needed, bring insurance card
- High-dose available for 65+

WEIGHT WATCHERS AT WORK
- New session starts November 1st
- Meetings Wednesdays at noon, Finance Conference Room
- City subsidizes 50% of membership
- Minimum 12 participants needed

MEDITATION MONDAYS
- 15-minute guided sessions
- Parks Building break room, 12:15pm
- Drop-in format, no registration

Remember: Wellness program participants earn a $20/month insurance premium discount. Complete 3 activities per quarter to maintain the discount.

Stay healthy!
Wellness Committee""",
        "has_attachment": False,
        "attachment_names": []
    },

    # IT/Technology Emails (295-302)
    {
        "id": "email_295",
        "thread_id": None,
        "thread_position": None,
        "thread_length": None,
        "date": "2023-01-18T07:00:00",
        "from": "it-helpdesk@cityofexample.gov",
        "to": ["all-staff@cityofexample.gov"],
        "cc": [],
        "subject": "URGENT: Password Reset Required by January 31",
        "body": """All Users,

Per our updated security policy, all network passwords must be reset before January 31st.

This is a mandatory requirement for continued system access.

New password requirements:
- Minimum 12 characters (increased from 8)
- At least one uppercase letter
- At least one lowercase letter
- At least one number
- At least one special character (!@#$%^&*)
- Cannot reuse last 10 passwords
- Cannot contain your username or name

How to reset:
1. Press Ctrl+Alt+Delete
2. Select "Change a password"
3. Enter current password, then new password twice
4. Click the arrow to submit

If you're having trouble:
- Call the Help Desk: ext. 4357 (HELP)
- Submit a ticket: helpdesk.cityofexample.gov
- Visit IT in Room 101 (City Hall basement)

Accounts that haven't reset by January 31st will be locked and require an in-person visit with photo ID to unlock.

Thank you for helping keep our systems secure.

IT Security Team""",
        "has_attachment": False,
        "attachment_names": []
    },
    {
        "id": "email_296",
        "thread_id": None,
        "thread_position": None,
        "thread_length": None,
        "date": "2023-02-28T17:30:00",
        "from": "it@cityofexample.gov",
        "to": ["all-staff@cityofexample.gov"],
        "cc": [],
        "subject": "Email System Maintenance - Saturday 3/4",
        "body": """All Staff,

Scheduled maintenance on our email system this weekend:

WHEN: Saturday, March 4th, 10 PM - Sunday, March 5th, 6 AM

IMPACT:
- Outlook/email will be unavailable
- Calendar invites won't sync
- Email on phones won't update
- Webmail (OWA) will be down

WHAT'S HAPPENING:
- Migrating to new Exchange servers
- Storage capacity doubled
- Improved spam filtering
- Faster search capabilities

WHAT TO DO:
- Save any drafts before Saturday evening
- Download important attachments you may need
- If urgent, use personal email or phone for emergencies

AFTER MAINTENANCE:
- Outlook should reconnect automatically
- May need to re-enter password on mobile devices
- Archive folders may take a few hours to fully appear

If issues persist after 8 AM Sunday, contact the Help Desk.

We apologize for any inconvenience. This upgrade will significantly improve system performance.

IT Department""",
        "has_attachment": False,
        "attachment_names": []
    },
    {
        "id": "email_297",
        "thread_id": None,
        "thread_position": None,
        "thread_length": None,
        "date": "2023-03-20T10:00:00",
        "from": "it-security@cityofexample.gov",
        "to": ["all-staff@cityofexample.gov"],
        "cc": [],
        "subject": "Phishing Alert - Tax Season Scams",
        "body": """All Staff,

We've seen an increase in phishing attempts targeting city employees. Several recent examples:

FAKE IRS EMAILS
- Subject: "Your tax refund is ready"
- Claims you need to "verify" information to receive refund
- Links go to fake IRS website
- THE IRS DOES NOT EMAIL YOU

FAKE W-2 REQUESTS
- Appears to come from HR or Finance
- Asks for copies of W-2s or tax documents
- Often targets payroll/HR staff
- ALWAYS verify by phone before sending sensitive docs

FAKE GIFT CARD REQUESTS
- Appears to come from a manager or executive
- "Urgent" request to buy gift cards
- Asks you to send the card numbers
- NEVER purchase gift cards on behalf of anyone

How to identify phishing:
- Hover over links before clicking (does URL match?)
- Check sender address carefully (smith@city0fexample.gov vs @cityofexample.gov)
- Urgency + secrecy = red flag
- When in doubt, call the supposed sender directly

If you clicked a suspicious link:
1. Disconnect from network immediately
2. Call IT Security: ext. 4277
3. Change your password from a different device

Report suspicious emails: forward to phishing@cityofexample.gov

IT Security""",
        "has_attachment": False,
        "attachment_names": []
    },
    {
        "id": "email_298",
        "thread_id": None,
        "thread_position": None,
        "thread_length": None,
        "date": "2023-04-05T14:15:00",
        "from": "it@cityofexample.gov",
        "to": ["all-staff@cityofexample.gov"],
        "cc": [],
        "subject": "New Laptop Deployment - Schedule Your Appointment",
        "body": """All Staff,

Good news! We're replacing all laptops that are 5+ years old with new Dell Latitude models.

WHO'S ELIGIBLE:
If you received an email last week with your asset tag number, you're on the list.

WHAT'S NEW:
- Windows 11 Pro
- Intel Core i7 processor
- 16GB RAM (up from 8GB)
- 512GB SSD (up from 256GB)
- Improved battery life (up to 10 hours)
- Lighter weight

HOW TO SCHEDULE:
1. Visit helpdesk.cityofexample.gov/laptop-upgrade
2. Select an available time slot (30 min appointments)
3. Location: IT Office, City Hall Room 101

WHAT TO DO BEFORE YOUR APPOINTMENT:
- Back up personal files to OneDrive
- Note any special software you use
- Make list of printers you connect to
- Bring your old laptop AND power cord

We'll transfer your files and settings. Most apps install automatically; specialized software may take an extra day.

Deployment schedule:
- April 10-28: City Hall
- May 1-19: Public Works
- May 22-June 9: Police/Fire
- June 12-30: All other locations

IT Department""",
        "has_attachment": False,
        "attachment_names": []
    },
    {
        "id": "email_299",
        "thread_id": None,
        "thread_position": None,
        "thread_length": None,
        "date": "2023-05-22T09:30:00",
        "from": "it@cityofexample.gov",
        "to": ["all-staff@cityofexample.gov"],
        "cc": [],
        "subject": "Microsoft Teams Training - Getting Started",
        "body": """All Staff,

As we transition from Skype to Microsoft Teams, please take advantage of these training opportunities:

LIVE TRAINING SESSIONS (all via Teams, ironically):

Beginner - "Teams Basics"
- Tues 5/30, 10am or Wed 5/31, 2pm
- Covers: Chat, calls, meetings, files

Intermediate - "Effective Meetings"
- Tues 6/6, 10am or Wed 6/7, 2pm
- Covers: Scheduling, screen sharing, recording, backgrounds

Advanced - "Channels & Collaboration"
- Tues 6/13, 10am or Wed 6/14, 2pm
- Covers: Team creation, channels, tabs, apps

SELF-PACED RESOURCES:
- Microsoft Learn modules (free) - link on intranet
- Quick reference guides posted to IT SharePoint
- Video tutorials in LMS under "Software Training"

IMPORTANT DATES:
- June 1: Teams available to all staff
- June 30: Skype for Business disabled
- July 1: Teams only

Tip: Download the Teams mobile app for calls and chats on the go.

Questions? Attend our weekly "Teams Office Hours" - Thursdays 11-12, IT Training Room.

IT Department""",
        "has_attachment": True,
        "attachment_names": ["Teams_Quick_Reference.pdf"]
    },
    {
        "id": "email_300",
        "thread_id": None,
        "thread_position": None,
        "thread_length": None,
        "date": "2023-07-17T15:45:00",
        "from": "it-helpdesk@cityofexample.gov",
        "to": ["all-staff@cityofexample.gov"],
        "cc": [],
        "subject": "Network Slowness - Issue Resolved",
        "body": """All Staff,

The network performance issues reported this morning have been resolved.

WHAT HAPPENED:
At approximately 9:15 AM, users began experiencing slow network performance, including:
- Slow file access
- Email delays
- Application timeouts
- Video meeting issues

ROOT CAUSE:
A misconfigured network switch at our main distribution center was causing packet loss. The switch was replaced and properly configured.

RESOLUTION:
- Issue identified: 10:30 AM
- Switch replaced: 12:45 PM
- Full service restored: 1:15 PM

WHAT YOU MAY NEED TO DO:
- If you're still experiencing issues, try restarting your computer
- If Outlook shows "disconnected," it should reconnect within 10 minutes
- VPN users: disconnect and reconnect

We apologize for the disruption. If you continue to experience problems, please contact the Help Desk.

Thank you for your patience and for reporting the issue quickly - that helped us diagnose the problem faster.

IT Help Desk""",
        "has_attachment": False,
        "attachment_names": []
    },
    {
        "id": "email_301",
        "thread_id": None,
        "thread_position": None,
        "thread_length": None,
        "date": "2023-08-14T11:00:00",
        "from": "gis@cityofexample.gov",
        "to": ["department-heads@cityofexample.gov"],
        "cc": ["it@cityofexample.gov"],
        "subject": "New GIS Portal Launching September 1",
        "body": """Department Heads,

Our new public-facing GIS portal launches September 1st, replacing the aging system we've used since 2015.

NEW FEATURES:
- Modern, mobile-friendly interface
- Faster map loading
- Enhanced search functionality
- Integration with permitting system
- Downloadable data layers
- Street-level imagery

AVAILABLE MAP LAYERS:
- Zoning and land use
- Parcel boundaries with owner info
- City facilities and parks
- Council district boundaries
- Census demographics
- Business licenses (geocoded)
- Crime statistics (public version)

INTERNAL VERSION:
Staff will have access to additional layers not available publicly, including underground utilities and infrastructure details.

TRAINING:
- Public portal overview: Aug 28, 2pm
- Internal features deep-dive: Aug 30, 10am
- Location: Council Chambers

Please share with your teams. The portal will be a great resource for residents and staff alike.

GIS Coordinator""",
        "has_attachment": False,
        "attachment_names": []
    },
    {
        "id": "email_302",
        "thread_id": None,
        "thread_position": None,
        "thread_length": None,
        "date": "2023-10-16T08:00:00",
        "from": "it@cityofexample.gov",
        "to": ["all-staff@cityofexample.gov"],
        "cc": [],
        "subject": "Cybersecurity Awareness Month - Training Requirement",
        "body": """All Staff,

October is Cybersecurity Awareness Month. All employees must complete annual security awareness training by October 31st.

THIS YEAR'S COURSE COVERS:
- Password security and MFA
- Phishing and social engineering
- Physical security (tailgating, clean desk)
- Mobile device security
- Working securely from home
- Ransomware prevention
- Reporting incidents

HOW TO COMPLETE:
1. Log into the LMS (learning.cityofexample.gov)
2. Select "Cybersecurity Awareness 2023"
3. Complete all modules (~45 minutes total)
4. Pass the final quiz (80% required)

Those who completed training in September for the early-bird campaign are already done.

WHY IT MATTERS:
Last year, we blocked:
- 2.3 million spam emails
- 45,000 phishing attempts
- 1,200 malware attacks
But we're only as secure as our weakest link. One click on a bad link can compromise everything.

Incentive: Complete by October 25th for a chance to win one of ten $50 Amazon gift cards.

IT Security Team""",
        "has_attachment": False,
        "attachment_names": []
    },

    # General Administrative (303-310)
    {
        "id": "email_303",
        "thread_id": None,
        "thread_position": None,
        "thread_length": None,
        "date": "2023-01-23T14:00:00",
        "from": "facilities@cityofexample.gov",
        "to": ["city-hall-staff@cityofexample.gov"],
        "cc": [],
        "subject": "Parking Permit Renewals - 2023",
        "body": """City Hall Staff,

Parking permit renewals for 2023 are now open.

RENEWAL PROCESS:
1. Complete the online form (link on intranet)
2. Submit by February 15th
3. Pick up new permit from Security desk starting February 20th

FEES (payroll deduction available):
- Surface lot: $30/month
- Covered structure: $50/month
- Reserved spot: $75/month (waitlist only)

CHANGES THIS YEAR:
- New access cards for garage (old ones expire March 1)
- Electric vehicle charging spots expanded (4 new stations)
- Motorcycle parking moved to Row J
- Bicycle lockers now available ($10/month)

WAITLIST:
The covered structure has a 6-month waitlist. To add your name, check the box on the renewal form.

CARPOOL DISCOUNT:
Register a carpool of 3+ employees for 25% off all participants' fees.

Return your 2022 parking sticker when picking up your new permit.

Facilities Management""",
        "has_attachment": False,
        "attachment_names": []
    },
    {
        "id": "email_304",
        "thread_id": None,
        "thread_position": None,
        "thread_length": None,
        "date": "2023-02-06T10:30:00",
        "from": "clerk@cityofexample.gov",
        "to": ["department-admins@cityofexample.gov"],
        "cc": [],
        "subject": "Records Retention - Annual Purge Reminder",
        "body": """Department Administrators,

It's time for our annual records review and purge.

Per the Records Retention Schedule, many record types can now be destroyed if they've passed their retention period.

COMMON RETENTION PERIODS:
- Routine correspondence: 2 years
- Financial records: 7 years
- Personnel files: 7 years after separation
- Meeting agendas/minutes: Permanent (do NOT destroy)
- Contracts: 10 years after expiration
- Emails: Follow same schedule as paper records

DESTRUCTION PROCESS:
1. Review your department's records
2. Complete the Records Destruction Authorization form
3. Submit to City Clerk for approval
4. Approved boxes go to Iron Mountain for secure shredding

DEADLINES:
- Forms due: March 15th
- Shredding pickup: March 28-29th

REMINDERS:
- When in doubt, keep it
- Check with City Attorney if litigation might be pending
- Don't forget electronic records (shared drives, email archives)
- Public records requests can halt destruction

Training session: February 21st, 2pm, Room 201
Bring questions about your specific records.

City Clerk's Office""",
        "has_attachment": True,
        "attachment_names": ["Retention_Schedule_2023.pdf", "Destruction_Auth_Form.pdf"]
    },
    {
        "id": "email_305",
        "thread_id": None,
        "thread_position": None,
        "thread_length": None,
        "date": "2023-03-08T09:15:00",
        "from": "supplies@cityofexample.gov",
        "to": ["department-admins@cityofexample.gov"],
        "cc": [],
        "subject": "Office Supply Ordering - New System",
        "body": """Department Administrators,

We're switching to a new office supply ordering system starting April 1st.

NEW SYSTEM: OfficeMax Business Solutions
- Online ordering portal
- Next-day delivery on most items
- Better pricing through consortium contract
- Environmental preferred products highlighted

HOW TO ORDER:
1. Go to business.officemax.com
2. Log in with your city email (accounts being created now)
3. Search or browse products
4. Add to cart and checkout
5. Select your delivery location

APPROVAL WORKFLOW:
- Orders under $100: Auto-approved
- Orders $100-500: Department admin approval
- Orders over $500: Director approval required

TRAINING:
March 27th, 10am or 2pm, Training Room B
30-minute overview of the new system

NOTE: Use up existing Staples supplies before ordering new.
Final Staples order deadline: March 22nd.

Questions? Contact supplies@cityofexample.gov

Purchasing Department""",
        "has_attachment": False,
        "attachment_names": []
    },
    {
        "id": "email_306",
        "thread_id": None,
        "thread_position": None,
        "thread_length": None,
        "date": "2023-04-17T13:30:00",
        "from": "city-manager@cityofexample.gov",
        "to": ["all-staff@cityofexample.gov"],
        "cc": [],
        "subject": "Casual Friday - Updated Dress Code",
        "body": """All Staff,

In response to employee feedback, we're updating our dress code policy for Fridays.

EFFECTIVE MAY 1ST:
Casual Friday dress code will be relaxed to allow jeans (in good condition) for all departments.

STILL NOT PERMITTED:
- Ripped or torn jeans
- Shorts
- T-shirts with logos/graphics (plain is OK)
- Flip-flops or beach sandals
- Athletic wear

EXCEPTIONS:
Employees with scheduled public meetings, court appearances, or formal events should dress appropriately for those occasions regardless of day.

Field employees already under different dress code provisions are not affected.

This change is a trial through the end of the year. If we maintain our professional image, it will become permanent.

Thank you to the Employee Committee for bringing this forward.

Mark Thompson
City Manager""",
        "has_attachment": False,
        "attachment_names": []
    },
    {
        "id": "email_307",
        "thread_id": None,
        "thread_position": None,
        "thread_length": None,
        "date": "2023-05-30T11:00:00",
        "from": "security@cityofexample.gov",
        "to": ["city-hall-staff@cityofexample.gov"],
        "cc": [],
        "subject": "Building Access - New Badge Readers",
        "body": """City Hall Staff,

New badge readers are being installed this week. Here's what you need to know:

INSTALLATION SCHEDULE:
- Mon 6/5: Main entrance (will use temporary reader)
- Tue 6/6: Employee entrance
- Wed 6/7: Interior doors (Finance, HR, IT)
- Thu 6/8: Garage access points

YOUR CURRENT BADGE WILL WORK - no action needed.

NEW FEATURES:
- Faster read time
- Works from farther away (2 feet vs 2 inches)
- LED indicators (green = access, red = denied, yellow = try again)
- Mobile badge option coming soon

LOST/STOLEN BADGES:
Report immediately to Security: ext. 4111
Badges can now be deactivated within minutes

TAILGATING REMINDER:
Each person must badge in separately. Holding doors for unknown individuals creates security risks. It's OK to politely ask someone to use their badge.

Questions? Contact Security at ext. 4111 or security@cityofexample.gov

Security Department""",
        "has_attachment": False,
        "attachment_names": []
    },
    {
        "id": "email_308",
        "thread_id": None,
        "thread_position": None,
        "thread_length": None,
        "date": "2023-06-26T15:00:00",
        "from": "fleet@cityofexample.gov",
        "to": ["vehicle-operators@cityofexample.gov"],
        "cc": [],
        "subject": "City Vehicle Policy Reminder",
        "body": """City Vehicle Operators,

As we head into summer travel season, a few reminders about city vehicle use:

AUTHORIZED USE:
- City business only
- De minimis personal use OK (coffee stop, lunch)
- No passengers unless city business-related
- No pets

FUEL:
- Use Voyager fleet card at any major station
- No premium gas unless required (check vehicle manual)
- Report lost/stolen cards immediately

MAINTENANCE:
- Report issues promptly via Fleet portal
- Don't ignore warning lights
- Check tire pressure monthly
- Interior cleanliness is operator responsibility

ACCIDENTS:
1. Ensure safety first
2. Call police for any injury or significant damage
3. Exchange information with other parties
4. Take photos
5. Notify your supervisor immediately
6. Complete accident report within 24 hours

CELL PHONES:
Per state law and city policy, no handheld use while driving.
Hands-free is permitted for work calls.
Pull over for anything complex.

Drive safely out there!

Fleet Manager""",
        "has_attachment": False,
        "attachment_names": []
    },
    {
        "id": "email_309",
        "thread_id": None,
        "thread_position": None,
        "thread_length": None,
        "date": "2023-08-21T10:45:00",
        "from": "mailroom@cityofexample.gov",
        "to": ["all-staff@cityofexample.gov"],
        "cc": [],
        "subject": "Mail Services - Updated Delivery Schedule",
        "body": """All Staff,

Starting September 1st, internal mail delivery schedules are changing:

NEW SCHEDULE:
- City Hall: 10am and 2pm (previously 9am, 12pm, 3pm)
- Public Works: 11am only (previously 10am, 2pm)
- Police/Fire: 1pm only (previously varies)
- Parks facilities: Tuesdays and Thursdays

WHAT THIS MEANS:
- Fewer but more reliable delivery times
- Larger capacity mail carts
- Same-day delivery if dropped by 1pm

OUTGOING MAIL:
- USPS pickup: 4pm daily
- FedEx/UPS pickup: 3pm daily
- Place in outgoing bin by pickup time

PACKAGES:
- Large packages: Pick up from mailroom
- Notification email sent when package arrives
- Please retrieve within 3 days

INTEROFFICE ENVELOPES:
Please reuse! Cross out previous names/departments.
New envelopes available from supply room.

Questions? Contact mailroom@cityofexample.gov or ext. 4200

Mailroom Services""",
        "has_attachment": False,
        "attachment_names": []
    },
    {
        "id": "email_310",
        "thread_id": None,
        "thread_position": None,
        "thread_length": None,
        "date": "2023-10-30T09:00:00",
        "from": "facilities@cityofexample.gov",
        "to": ["city-hall-staff@cityofexample.gov"],
        "cc": [],
        "subject": "Restroom Renovation - 2nd Floor",
        "body": """City Hall Staff,

The 2nd floor restrooms will be closed for renovation beginning November 6th.

PROJECT TIMELINE:
- November 6-17: Women's restroom
- November 20-December 1: Men's restroom
- December 4: Both reopen

ALTERNATIVE FACILITIES:
- 1st floor restrooms (near Council Chambers)
- 3rd floor restrooms (Finance wing)
- Accessible restroom on 1st floor remains available

WHAT'S BEING UPDATED:
- New fixtures (low-flow toilets and faucets)
- LED lighting
- Tile replacement
- Fresh paint
- Baby changing stations in both restrooms

We apologize for the inconvenience. These updates are long overdue and will improve everyone's experience.

Contractors will work 7am-4pm to minimize disruption. Please expect some noise.

Facilities Management""",
        "has_attachment": False,
        "attachment_names": []
    },

    # Finance/Accounting (311-318)
    {
        "id": "email_311",
        "thread_id": None,
        "thread_position": None,
        "thread_length": None,
        "date": "2023-01-04T08:30:00",
        "from": "payroll@cityofexample.gov",
        "to": ["all-staff@cityofexample.gov"],
        "cc": [],
        "subject": "2023 Payroll Calendar and W-2 Information",
        "body": """All Employees,

Happy New Year! A few payroll items to start 2023:

2023 PAY DATES:
Attached is the pay calendar showing all 26 pay dates for the year.
Direct deposit hits Thursday night; checks available Friday morning.

W-2 FORMS:
- Mailed by January 31st to address on file
- Electronic W-2s available in the portal starting January 15th
- Opt into electronic delivery for faster access

VERIFY YOUR INFORMATION:
Log into the employee portal to confirm:
- Current mailing address
- W-4 withholding elections
- Direct deposit accounts
- Emergency contacts

TAX CHANGES FOR 2023:
- Social Security wage base increased to $160,200
- CA SDI rate: 1.1% (up from 1.0%)
- 401(a) contribution limit: $66,000
- 457(b) contribution limit: $22,500

PENSION CONTRIBUTIONS:
CalPERS rates remain unchanged for most tiers.
See attached memo if you're in the PEPRA tier.

Questions? Contact payroll@cityofexample.gov

Payroll Department""",
        "has_attachment": True,
        "attachment_names": ["2023_Pay_Calendar.pdf", "PEPRA_Memo.pdf"]
    },
    {
        "id": "email_312",
        "thread_id": None,
        "thread_position": None,
        "thread_length": None,
        "date": "2023-02-10T14:00:00",
        "from": "finance@cityofexample.gov",
        "to": ["department-heads@cityofexample.gov"],
        "cc": [],
        "subject": "FY24 Budget Development - Timeline and Instructions",
        "body": """Department Heads,

Budget season is upon us. Here's what to expect:

FY2024 BUDGET TIMELINE:
- Feb 15: Budget kickoff meeting (mandatory)
- Feb 28: Preliminary revenue projections distributed
- Mar 15: Department budget requests due
- Apr 1-15: Finance review and questions
- Apr 17-28: Department hearings with City Manager
- May 15: Proposed budget to Council
- June 1: Public hearing
- June 15: Budget adoption

BUDGET GUIDANCE:
Given current projections, we're asking departments to prepare two scenarios:
1. Flat budget (0% increase over current year)
2. Priority-based reduction (2% reduction with ranked priorities)

NEW THIS YEAR:
- Online budget portal (no more spreadsheets!)
- Training scheduled Feb 22nd
- Position requests submitted separately

CAPITAL REQUESTS:
Submit via the CIP portal by March 1st.
Threshold remains $50,000 for capital classification.

Please confirm attendance at the kickoff meeting.

Robert Chen
Finance Director""",
        "has_attachment": False,
        "attachment_names": []
    },
    {
        "id": "email_313",
        "thread_id": None,
        "thread_position": None,
        "thread_length": None,
        "date": "2023-03-15T11:30:00",
        "from": "accounts-payable@cityofexample.gov",
        "to": ["all-staff@cityofexample.gov"],
        "cc": [],
        "subject": "P-Card Policy Reminder and Common Violations",
        "body": """Purchasing Card Holders,

A few reminders as we complete our annual P-Card audit:

COMMON VIOLATIONS (please avoid!):
1. Missing receipts (most common - 45% of issues)
2. Sales tax paid when exempt (we're tax-exempt!)
3. Split transactions to avoid approval limits
4. Personal purchases (even if reimbursed later)
5. Late reconciliation

RECEIPTS:
- Upload within 5 business days of transaction
- Photo receipts are fine but must be legible
- If receipt lost, complete the Missing Receipt form

SALES TAX:
We are exempt from CA sales tax. If charged:
- Ask merchant to credit back
- If unable, provide our exemption certificate
- Note on receipt that exemption was attempted

TRANSACTION LIMITS:
- Single purchase: $2,500
- Monthly limit: $5,000
- Do NOT split purchases to stay under limits

REVIEW CYCLE:
- Transactions post within 3 days
- Reconcile weekly in the portal
- Supervisor approval due by the 5th of following month

Cards with chronic violations may be revoked.

Accounts Payable""",
        "has_attachment": True,
        "attachment_names": ["Tax_Exempt_Certificate.pdf"]
    },
    {
        "id": "email_314",
        "thread_id": None,
        "thread_position": None,
        "thread_length": None,
        "date": "2023-04-24T16:45:00",
        "from": "finance@cityofexample.gov",
        "to": ["all-staff@cityofexample.gov"],
        "cc": [],
        "subject": "Expense Reimbursement - New Mileage Rate",
        "body": """All Staff,

Effective April 1st, the IRS mileage reimbursement rate increased:

NEW RATE: $0.655 per mile (up from $0.625)

This applies to:
- Personal vehicle use for city business
- Travel between work sites
- Training and conference travel

REMINDER: Mileage is calculated from your regular work site, not your home, unless otherwise approved.

SUBMITTING CLAIMS:
1. Complete mileage log with:
   - Date
   - Starting and ending locations
   - Business purpose
   - Calculated miles (use Google Maps)
2. Attach log to expense report in portal
3. Submit within 30 days of travel

COMMUTE MILEAGE:
Normal commute is NOT reimbursable.
Traveling from home directly to a training? Subtract your normal commute distance.

PARKING AND TOLLS:
Reimbursable with receipts. Attach to expense report.

Questions? Contact accounts-payable@cityofexample.gov

Finance Department""",
        "has_attachment": False,
        "attachment_names": []
    },
    {
        "id": "email_315",
        "thread_id": None,
        "thread_position": None,
        "thread_length": None,
        "date": "2023-05-08T09:00:00",
        "from": "finance@cityofexample.gov",
        "to": ["department-heads@cityofexample.gov"],
        "cc": [],
        "subject": "FY23 Year-End Close - Action Required",
        "body": """Department Heads,

As we approach fiscal year end (June 30), please ensure the following:

PURCHASE ORDERS:
- Review all open POs by June 9
- Cancel POs that won't be fulfilled
- Receive goods in system promptly
- Encumbered funds not used will lapse

INVOICES:
- Submit all FY23 invoices by June 16
- Mark clearly "FY23" on invoice transmittals
- Late invoices may be charged to FY24 budget

CONTRACTS:
- Final deliverables received by June 23
- Ensure all contract work is complete
- Notify Finance of any amendments needed

CAPITAL PROJECTS:
- Update project status in portal
- Submit any budget transfer requests by June 9
- Carryforward requests due June 16

YEAR-END PROCESSING:
- Last check run for FY23: June 27
- Emergency payments only after June 27
- Travel reimbursements: submit by June 16

Detailed year-end memo attached. Please share with your fiscal staff.

Finance Department""",
        "has_attachment": True,
        "attachment_names": ["FY23_Year_End_Memo.pdf"]
    },
    {
        "id": "email_316",
        "thread_id": None,
        "thread_position": None,
        "thread_length": None,
        "date": "2023-07-03T10:15:00",
        "from": "finance@cityofexample.gov",
        "to": ["department-heads@cityofexample.gov"],
        "cc": [],
        "subject": "FY24 Budget Approved - Account Access Now Available",
        "body": """Department Heads,

Council adopted the FY2024 budget on June 20th. Here's what you need to know:

BUDGET HIGHLIGHTS:
- General Fund: $127.4M (up 3.2%)
- Total All Funds: $198.6M
- 12 new positions approved
- 4.5% COLA for represented employees (per MOU)

YOUR DEPARTMENT:
Individual budget detail is now available in the budget portal. Please review your allocations and contact Finance if you have questions.

NEW POSITIONS:
If you received new headcount approval, coordinate with HR to begin recruitment.

PURCHASING:
- FY24 requisitions can now be entered
- Review recurring contracts for renewals
- Capital project spending may begin

BUDGET AMENDMENTS:
Council approved two mid-year review dates:
- December 12 (Q1 review)
- March 19 (Q2 review)
Prepare amendment requests at least 30 days before.

QUESTIONS:
Budget analyst assignments attached. Your analyst is your first point of contact.

Thank you for your work during budget development!

Finance Director""",
        "has_attachment": True,
        "attachment_names": ["Budget_Analyst_Assignments.pdf"]
    },
    {
        "id": "email_317",
        "thread_id": None,
        "thread_position": None,
        "thread_length": None,
        "date": "2023-09-11T14:30:00",
        "from": "auditor@cityofexample.gov",
        "to": ["department-heads@cityofexample.gov"],
        "cc": ["finance@cityofexample.gov"],
        "subject": "Annual Audit - Information Request",
        "body": """Department Heads,

Our external auditors (Smith & Associates) will begin FY2023 audit fieldwork on October 2nd.

INFORMATION NEEDED BY SEPTEMBER 29:

All Departments:
- Outstanding receivables as of June 30
- Significant contracts entered/completed in FY23
- Known contingencies or litigation
- Federal/state grant documentation

Finance:
- Compilation support in progress

Public Works:
- Fixed asset additions/disposals
- CIP status reports

Police/Fire:
- Grant expenditure documentation
- Asset forfeiture activity

AUDIT SCHEDULE:
- Oct 2-6: Planning and internal controls
- Oct 9-20: Substantive testing
- Oct 23-27: Wrap-up and management letter
- Nov 15: Draft report to Finance/Audit Committee
- Dec 5: Final report to Council

Please make staff available for auditor questions during fieldwork.

Contact me with any concerns about the audit scope or timing.

City Auditor""",
        "has_attachment": False,
        "attachment_names": []
    },
    {
        "id": "email_318",
        "thread_id": None,
        "thread_position": None,
        "thread_length": None,
        "date": "2023-11-15T11:00:00",
        "from": "treasury@cityofexample.gov",
        "to": ["department-admins@cityofexample.gov"],
        "cc": [],
        "subject": "Petty Cash - Annual Verification Required",
        "body": """Petty Cash Custodians,

Time for our annual petty cash verification!

BY NOVEMBER 30, please:

1. Count your petty cash fund
2. Ensure count + receipts = authorized amount
3. Complete the Petty Cash Verification form
4. Have your supervisor sign
5. Return form to Treasury

COMMON ISSUES:
- Receipts not replenished (submit for reimbursement!)
- IOUs (not permitted)
- Overages/shortages (investigate and document)

REMINDERS:
- Maximum single disbursement: $50
- Receipts required for all disbursements
- Replenish when fund reaches 25% of authorized amount
- No check cashing from petty cash

FUND CHANGES:
If your department no longer needs petty cash, or needs the amount adjusted, indicate on the form. We'll schedule a pickup or adjustment.

Departments without petty cash: no action needed.

Questions? Contact treasury@cityofexample.gov

Treasury Division""",
        "has_attachment": True,
        "attachment_names": ["Petty_Cash_Verification_Form.pdf"]
    },

    # Events/Community (319-324)
    {
        "id": "email_319",
        "thread_id": None,
        "thread_position": None,
        "thread_length": None,
        "date": "2023-04-10T10:00:00",
        "from": "employee-committee@cityofexample.gov",
        "to": ["all-staff@cityofexample.gov"],
        "cc": [],
        "subject": "Annual Employee Picnic - Save the Date!",
        "body": """City Employees,

Mark your calendars for the annual Employee Appreciation Picnic!

WHEN: Saturday, June 10th, 11am-3pm
WHERE: Riverside Park, Pavilion A

WHAT'S INCLUDED:
- BBQ lunch (burgers, hot dogs, veggie options)
- Drinks and dessert
- Games and activities for all ages
- DJ and dancing
- Raffle prizes

FAMILY WELCOME:
Employees may bring immediate family members at no cost.
Please RSVP with headcount by June 1st.

VOLUNTEERS NEEDED:
We're looking for help with:
- Setup (9-11am)
- Grilling (11am-1pm)
- Games coordination (11am-2pm)
- Cleanup (2-4pm)

Sign up to volunteer and get entered for special door prizes!

RSVP: picnic.cityofexample.gov
Volunteer signup: Same link, check the box

This is a great chance to connect with colleagues outside the office. Hope to see you there!

Employee Events Committee""",
        "has_attachment": False,
        "attachment_names": []
    },
    {
        "id": "email_320",
        "thread_id": None,
        "thread_position": None,
        "thread_length": None,
        "date": "2023-05-01T09:30:00",
        "from": "wellness@cityofexample.gov",
        "to": ["all-staff@cityofexample.gov"],
        "cc": [],
        "subject": "American Red Cross Blood Drive - May 17",
        "body": """City Staff,

Please consider donating blood at our on-site blood drive!

DATE: Wednesday, May 17th
TIME: 9am - 2pm
LOCATION: City Hall, Conference Room A

WHY DONATE:
- One donation can save up to 3 lives
- Blood supply is critically low this spring
- 1 hour of your time makes a real difference

ELIGIBILITY:
- Generally healthy and feeling well
- At least 17 years old (16 with parental consent)
- Weigh at least 110 lbs
- Have not donated in the last 56 days

SCHEDULING:
Sign up for a time slot: redcrossblood.org, sponsor code: CITYEXAMPLE

Same-day walk-ins accepted if appointments available.

AFTER DONATING:
Take your time in the refreshment area. If you feel lightheaded, stay as long as needed.

WORK TIME:
Donation time (approximately 1 hour) is considered work time with supervisor approval.

Every donation counts!

Wellness Committee""",
        "has_attachment": False,
        "attachment_names": []
    },
    {
        "id": "email_321",
        "thread_id": None,
        "thread_position": None,
        "thread_length": None,
        "date": "2023-08-07T14:00:00",
        "from": "employee-committee@cityofexample.gov",
        "to": ["all-staff@cityofexample.gov"],
        "cc": [],
        "subject": "United Way Campaign Kickoff",
        "body": """City Employees,

Our annual United Way campaign begins August 14th!

CAMPAIGN GOAL: $50,000

Last year, City employees raised $47,500 - our best year ever. Let's top it!

HOW TO GIVE:
- Payroll deduction (as little as $5/paycheck)
- One-time gift
- Designate to specific agencies if desired

SPECIAL EVENTS:
- Kickoff breakfast: Aug 14, 8am, Atrium
- Department competition: Highest participation wins pizza party
- Jeans Day: Donate $5 to wear jeans any Friday in August
- 50/50 raffle: Tickets $5 each, drawing Sept 1

WHERE YOUR MONEY GOES:
United Way supports local programs including:
- After-school youth programs
- Senior services
- Food banks
- Housing assistance
- Job training

LEADERSHIP GIVING:
Gifts of $500+ receive recognition at the campaign celebration.

Pledge forms available in your department or online: unitedway.cityofexample.gov

Every gift makes a difference!

United Way Campaign Committee""",
        "has_attachment": True,
        "attachment_names": ["UW_Pledge_Form.pdf"]
    },
    {
        "id": "email_322",
        "thread_id": None,
        "thread_position": None,
        "thread_length": None,
        "date": "2023-10-23T11:30:00",
        "from": "employee-committee@cityofexample.gov",
        "to": ["all-staff@cityofexample.gov"],
        "cc": [],
        "subject": "Thanksgiving Food Drive - Donations Needed",
        "body": """City Employees,

Help us fill the food bank shelves before Thanksgiving!

COLLECTION DATES: October 30 - November 17

DROP-OFF LOCATIONS:
- City Hall lobby (main collection point)
- Public Works breakroom
- Police Department front desk
- Parks & Rec headquarters

MOST NEEDED ITEMS:
- Canned vegetables and fruits
- Canned soups and stews
- Peanut butter
- Pasta and rice
- Cereal
- Cooking oil
- Stuffing mix and gravy

PLEASE AVOID:
- Expired items
- Glass containers
- Opened packages
- Items needing refrigeration

DEPARTMENT CHALLENGE:
The department that donates the most items per employee wins bragging rights and a trophy!

VOLUNTEER OPPORTUNITY:
Help sort donations at the food bank on November 18th, 9am-12pm. Sign up with your department admin.

Last year we collected over 3,000 items. Let's do even better!

Employee Committee""",
        "has_attachment": False,
        "attachment_names": []
    },
    {
        "id": "email_323",
        "thread_id": None,
        "thread_position": None,
        "thread_length": None,
        "date": "2023-11-20T10:00:00",
        "from": "city-manager@cityofexample.gov",
        "to": ["all-staff@cityofexample.gov"],
        "cc": [],
        "subject": "Holiday Party - December 15th",
        "body": """City Staff,

You're invited to our annual Holiday Celebration!

WHEN: Friday, December 15th, 6:00-9:00 PM
WHERE: Grand Ballroom, Marriott Downtown

DETAILS:
- Cocktails and hors d'oeuvres: 6:00-7:00
- Dinner and program: 7:00-8:30
- Dancing and dessert: 8:30-9:00

RSVP by December 8th at holiday.cityofexample.gov
Indicate meal preference (beef, chicken, fish, or vegetarian) and any dietary restrictions.

PLUS ONE:
Each employee may bring one guest at no charge.

DRESS CODE:
Festive cocktail attire

PROGRAM:
- City Manager's remarks
- Years of service recognition (5, 10, 15, 20, 25+ years)
- Retiree recognition
- Door prizes!

PARKING:
Validated at hotel garage

ALTERNATIVE:
If December 15th doesn't work for you, a lunchtime celebration will be held December 14th at City Hall.

Thank you for your dedicated service this year!

Mark Thompson
City Manager""",
        "has_attachment": False,
        "attachment_names": []
    },
    {
        "id": "email_324",
        "thread_id": None,
        "thread_position": None,
        "thread_length": None,
        "date": "2023-06-05T15:30:00",
        "from": "employee-committee@cityofexample.gov",
        "to": ["all-staff@cityofexample.gov"],
        "cc": [],
        "subject": "Softball Team - Players Needed!",
        "body": """Hey everyone,

The City of Example softball team is looking for players for the Municipal League summer season!

SEASON: July 6 - August 24 (8 weeks)
GAMES: Thursday evenings, 6:30 or 8:00 PM
LOCATION: Jefferson Field Complex

WHO CAN PLAY:
Current city employees and their adult family members (18+)

SKILL LEVEL:
Recreational - all skill levels welcome!
We're here to have fun, not win championships (though that would be nice too).

POSITIONS NEEDED:
- Outfielders (especially)
- Infielders
- Pitcher/catcher
- Designated hitters (if that's your thing)

COST:
$30 per player (covers league fees and team shirts)

SIGN UP:
Email softball@cityofexample.gov by June 23
Include your preferred position(s) and t-shirt size

Can't commit to the whole season? Let us know - subs welcome!

Last year was a blast. Join us!

Tom (Parks Dept) and Sarah (Finance)
Co-Team Captains""",
        "has_attachment": False,
        "attachment_names": []
    },

    # Facilities/Building (325-330)
    {
        "id": "email_325",
        "thread_id": None,
        "thread_position": None,
        "thread_length": None,
        "date": "2023-03-27T08:00:00",
        "from": "facilities@cityofexample.gov",
        "to": ["city-hall-staff@cityofexample.gov"],
        "cc": [],
        "subject": "HVAC Maintenance - Temporary Temperature Impacts",
        "body": """City Hall Staff,

Scheduled HVAC maintenance this week may cause temporary temperature fluctuations.

SCHEDULE:
- Monday: 1st floor
- Tuesday: 2nd floor
- Wednesday: 3rd floor
- Thursday: Mechanical room testing

WHAT TO EXPECT:
- HVAC may shut down for 30-60 minute periods
- Temperature may vary plus/minus 5 degrees during work
- Work performed 7am-4pm to minimize disruption

TIPS:
- Dress in layers
- Portable fans available from Facilities (call ext. 4500)
- Space heaters are NOT permitted (fire safety)
- Keep windows closed for system efficiency

This maintenance is necessary to ensure reliable cooling before summer.

Emergency after-hours: ext. 4911

Thank you for your patience.

Facilities Management""",
        "has_attachment": False,
        "attachment_names": []
    },
    {
        "id": "email_326",
        "thread_id": None,
        "thread_position": None,
        "thread_length": None,
        "date": "2023-04-03T14:15:00",
        "from": "facilities@cityofexample.gov",
        "to": ["city-hall-staff@cityofexample.gov"],
        "cc": [],
        "subject": "Annual Fire Drill - April 12",
        "body": """City Hall Staff,

Our annual fire drill is scheduled for:

DATE: Wednesday, April 12th
TIME: Between 10:00-11:00 AM (exact time unannounced)

WHEN THE ALARM SOUNDS:
1. Stop what you're doing
2. Do NOT use elevators
3. Exit via nearest stairwell
4. Proceed to assembly point (front parking lot)
5. Report to your department head for headcount
6. Wait for all-clear announcement

REMEMBER:
- Close doors behind you (don't lock)
- Take personal essentials (keys, phone, medication)
- Assist visitors and those needing help
- Stay with your department at assembly point

EVACUATION WARDENS:
Please review your floor assignments before the drill. Meet with me April 10th at 2pm for brief refresher.

MOBILITY ACCOMMODATIONS:
If you need assistance evacuating, please notify your supervisor in advance. We'll ensure a buddy is assigned.

All-clear should come within 15-20 minutes. Drill is required by fire code.

Facilities Manager""",
        "has_attachment": False,
        "attachment_names": []
    },
    {
        "id": "email_327",
        "thread_id": None,
        "thread_position": None,
        "thread_length": None,
        "date": "2023-05-15T11:45:00",
        "from": "facilities@cityofexample.gov",
        "to": ["city-hall-staff@cityofexample.gov"],
        "cc": [],
        "subject": "Elevator Inspection - Out of Service May 22-23",
        "body": """City Hall Staff,

The main elevator will be out of service for the state-required annual inspection.

DATES: Monday-Tuesday, May 22-23
BACKUP: Freight elevator available (slower, back of building)

AFFECTED:
- Main passenger elevator (lobby)
- Freight elevator remains operational

PLAN AHEAD:
If you have mobility challenges or heavy loads to move, please plan accordingly. Facilities staff can assist - call ext. 4500.

THE GOOD NEWS:
After inspection, the elevator control system will be upgraded. Expected improvements:
- Faster door response
- More accurate floor leveling
- Updated emergency phone

We'll be back to full service by Wednesday morning.

Thank you for your understanding.

Facilities Management""",
        "has_attachment": False,
        "attachment_names": []
    },
    {
        "id": "email_328",
        "thread_id": None,
        "thread_position": None,
        "thread_length": None,
        "date": "2023-07-24T09:30:00",
        "from": "facilities@cityofexample.gov",
        "to": ["all-city-buildings@cityofexample.gov"],
        "cc": [],
        "subject": "Pest Control Schedule - July/August",
        "body": """All City Buildings,

Monthly pest control treatment schedule:

JULY 27 (Thursday):
- City Hall: 6am-8am
- Parks HQ: 8am-10am

AUGUST 3 (Thursday):
- Public Works: 6am-8am
- Police HQ: 8am-10am

AUGUST 10 (Thursday):
- Fire Stations 1-3: 6am-8am
- Library: 8am-10am

WHAT TO EXPECT:
Treatment is applied along baseboards and in common areas. Products used are low-odor and safe once dry (approximately 30 minutes).

PLEASE BEFORE TREATMENT:
- Remove food from floor-level areas
- Close desk drawers
- Secure pet food if applicable (K-9 units)

IF YOU SEE PESTS:
Report immediately to facilities@cityofexample.gov with location and type. Photos helpful. We'll schedule targeted treatment.

FOOD STORAGE REMINDER:
- Keep food in sealed containers
- Clean up crumbs promptly
- Refrigerate or remove food waste daily
- Don't leave pet food overnight

Prevention is the best control!

Facilities Management""",
        "has_attachment": False,
        "attachment_names": []
    },
    {
        "id": "email_329",
        "thread_id": None,
        "thread_position": None,
        "thread_length": None,
        "date": "2023-09-18T13:00:00",
        "from": "facilities@cityofexample.gov",
        "to": ["city-hall-staff@cityofexample.gov"],
        "cc": [],
        "subject": "Cleaning Service Changes",
        "body": """City Hall Staff,

Starting October 1st, our cleaning contractor will make the following changes:

SCHEDULE CHANGES:
- Day porter now 7am-3pm (was 6am-2pm)
- Evening cleaning starts at 6pm (was 5:30pm)
- Saturday cleaning discontinued (budget)

SERVICE LEVELS:
Daily (unchanged):
- Restroom cleaning and restocking
- Trash removal
- Common area vacuuming
- Kitchen/breakroom cleaning

Weekly:
- Office vacuuming (Monday for odd floors, Wednesday for even)
- Dusting
- Glass doors and partitions

Monthly:
- Carpet deep clean (after hours)
- Window washing (exterior quarterly)

WHAT THIS MEANS FOR YOU:
If you work late, cleaning may occur while you're present. Crew will work around you.

CONCERNS OR ISSUES:
- Missed cleaning? Email facilities@cityofexample.gov
- Urgent spill? Call day porter at ext. 4555

Thank you for helping keep our building clean!

Facilities Management""",
        "has_attachment": False,
        "attachment_names": []
    },
    {
        "id": "email_330",
        "thread_id": None,
        "thread_position": None,
        "thread_length": None,
        "date": "2023-11-06T08:45:00",
        "from": "safety@cityofexample.gov",
        "to": ["all-staff@cityofexample.gov"],
        "cc": [],
        "subject": "Great California ShakeOut - November 9",
        "body": """All Staff,

This Thursday is the Great California ShakeOut earthquake drill!

DATE: Thursday, November 9th
TIME: 10:09 AM

WHEN THE SIGNAL SOUNDS:
Practice "Drop, Cover, and Hold On":
1. DROP to your hands and knees
2. Take COVER under a sturdy desk or table
3. HOLD ON until the shaking stops

IF NO DESK/TABLE:
- Cover your head and neck with your arms
- Move away from windows, shelves, heavy objects
- Stay where you are until shaking stops

AFTER "SHAKING" STOPS:
- Check yourself and others for injuries
- Evacuate if building damage visible
- Report to assembly point for headcount
- Await further instructions

FIELD EMPLOYEES:
If driving, pull over safely and stop.
If outdoors, move to open area away from buildings.

REVIEW:
- Know your building's evacuation routes
- Locate fire extinguishers
- Know where first aid kits are located
- Update your emergency contact info

California averages 10,000+ earthquakes per year. Be prepared!

Safety Officer""",
        "has_attachment": True,
        "attachment_names": ["Earthquake_Preparedness_Guide.pdf"]
    },

    # Legal/Risk (331-335)
    {
        "id": "email_331",
        "thread_id": None,
        "thread_position": None,
        "thread_length": None,
        "date": "2023-02-20T10:30:00",
        "from": "risk@cityofexample.gov",
        "to": ["supervisors@cityofexample.gov"],
        "cc": [],
        "subject": "Workers' Compensation - Injury Reporting Reminder",
        "body": """Supervisors,

A reminder on proper injury reporting procedures:

WHEN AN EMPLOYEE IS INJURED:
1. Ensure immediate medical attention if needed (call 911 for emergencies)
2. Complete Supervisor's Report within 24 hours
3. Employee completes Employee's Report
4. Submit both to Risk Management immediately

WHERE TO SUBMIT:
- Email: risk@cityofexample.gov
- Fax: 555-0102
- In person: City Hall Room 105

COMMON MISTAKES:
- Waiting too long to report (must be within 24 hours)
- Not documenting witness information
- Failing to preserve the scene for investigation
- Assuming minor injuries don't need reporting

ALL INJURIES MUST BE REPORTED regardless of severity. Even if the employee doesn't seek medical attention, document it.

RETURN TO WORK:
Modified duty may be available. Work with Risk Management and HR to accommodate restrictions.

QUESTIONS:
Contact Risk Management at ext. 4300

Training for new supervisors is available upon request.

Risk Management""",
        "has_attachment": True,
        "attachment_names": ["Supervisor_Injury_Report_Form.pdf"]
    },
    {
        "id": "email_332",
        "thread_id": None,
        "thread_position": None,
        "thread_length": None,
        "date": "2023-03-13T14:00:00",
        "from": "city-attorney@cityofexample.gov",
        "to": ["department-heads@cityofexample.gov"],
        "cc": [],
        "subject": "Contract Signature Authority Reminder",
        "body": """Department Heads,

A reminder on contract signature authority:

WHO CAN SIGN:
- City Manager: Contracts over $50,000 or any contract over 3 years
- Department Directors: Contracts up to $50,000, up to 3 years
- Division Managers: Contracts up to $25,000, up to 1 year

REQUIREMENTS BEFORE SIGNING:
1. Approved budget for expenditure
2. Insurance certificates on file with Risk
3. City Attorney review complete (required over $25,000)
4. Council approval if required by Municipal Code

COUNCIL APPROVAL REQUIRED FOR:
- Real property purchases/leases
- Revenue contracts
- Contracts over $100,000
- Multi-year contracts over $50,000 total
- Sole source contracts over $50,000

COMMON ISSUES WE'VE SEEN:
- Signing before legal review is complete
- Amending contracts without authorization
- Accepting vendor contract terms without negotiation
- Emergency procurements not properly documented

All executed contracts must be filed with the City Clerk within 5 days.

Questions? Contact the City Attorney's office.

City Attorney""",
        "has_attachment": True,
        "attachment_names": ["Signature_Authority_Chart.pdf"]
    },
    {
        "id": "email_333",
        "thread_id": None,
        "thread_position": None,
        "thread_length": None,
        "date": "2023-06-12T11:15:00",
        "from": "risk@cityofexample.gov",
        "to": ["department-heads@cityofexample.gov"],
        "cc": [],
        "subject": "Insurance Renewals - Information Needed by June 30",
        "body": """Department Heads,

Our liability insurance policies renew August 1st. The broker needs updated information:

NEEDED FROM EACH DEPARTMENT:

Vehicles:
- Updated vehicle list with VINs
- Any additions or deletions since last year
- Driver list for specialty vehicles

Property:
- New equipment over $10,000
- Renovations or additions to buildings
- Items stored off-site

Operations:
- New programs or services
- Special events planned for next year
- Any changes in scope of activities

Claims:
- Pending incidents not yet reported
- Near-misses worth documenting

DEADLINE: June 30th

Submit to risk@cityofexample.gov or schedule a call with me to walk through.

WHY IT MATTERS:
Accurate information ensures proper coverage. Unreported assets may not be covered if damaged. Underreported exposures can affect claims.

Our insurance costs are projected to increase 8-12% this year due to market conditions. Complete information helps us get the best rates.

Risk Manager""",
        "has_attachment": False,
        "attachment_names": []
    },
    {
        "id": "email_334",
        "thread_id": None,
        "thread_position": None,
        "thread_length": None,
        "date": "2023-08-28T09:45:00",
        "from": "city-attorney@cityofexample.gov",
        "to": ["all-staff@cityofexample.gov"],
        "cc": [],
        "subject": "Social Media Policy - Required Acknowledgment",
        "body": """All City Employees,

Per the recently updated Administrative Policy 7.3, all employees must acknowledge the City's social media policy by September 15th.

KEY POINTS:

Official Accounts:
- Only authorized staff may post on official City accounts
- All posts must be approved per department procedures
- Follow the social media style guide

Personal Accounts:
- You may identify yourself as a City employee
- You may NOT speak on behalf of the City
- Do not share confidential information
- Be respectful; inflammatory posts reflect on the City
- Do not use City resources (time, equipment) for personal social media

WHAT'S PROHIBITED:
- Posting non-public information
- Endorsing candidates or ballot measures as an employee
- Harassing or threatening content
- Content that would violate other policies

HOW TO ACKNOWLEDGE:
1. Log into the employee portal
2. Click "Policy Acknowledgments"
3. Read the full policy
4. Check the acknowledgment box

Failure to acknowledge by September 15th may result in restricted system access.

Questions? Contact the City Attorney's office.

City Attorney""",
        "has_attachment": True,
        "attachment_names": ["Policy_7.3_Social_Media.pdf"]
    },
    {
        "id": "email_335",
        "thread_id": None,
        "thread_position": None,
        "thread_length": None,
        "date": "2023-11-13T15:30:00",
        "from": "ethics@cityofexample.gov",
        "to": ["all-staff@cityofexample.gov"],
        "cc": [],
        "subject": "Holiday Gifts - Ethics Reminder",
        "body": """All City Employees,

As the holiday season approaches, a reminder about our gift policy:

WHAT YOU MAY ACCEPT:
- Gifts valued at $50 or less from a single source in a calendar year
- Food/perishables shared with the office
- Plaques, awards, or ceremonial items
- Gifts from personal friends/family unrelated to your position

WHAT YOU MAY NOT ACCEPT:
- Gifts over $50 from anyone doing business with the City
- Cash or cash equivalents (gift cards count as cash!)
- Gifts that could appear to influence your judgment
- Loans (other than from financial institutions at market rates)

REPORTING:
Gifts between $25-$50 should be reported to your supervisor.
Gifts over $50 must be declined or, if impractical, turned over to the City.

WHAT TO DO IF OFFERED INAPPROPRIATE GIFT:
- Politely decline, citing City policy
- If already received (e.g., delivered), notify your supervisor
- Don't feel bad - vendors know the rules

Remember: Even if technically allowed, consider the appearance. Would you be comfortable if it appeared in the newspaper?

Questions? Contact the Ethics Officer.

Ethics Office""",
        "has_attachment": False,
        "attachment_names": []
    },

    # Communications/Outreach (336-339)
    {
        "id": "email_336",
        "thread_id": None,
        "thread_position": None,
        "thread_length": None,
        "date": "2023-01-30T13:00:00",
        "from": "communications@cityofexample.gov",
        "to": ["department-pios@cityofexample.gov"],
        "cc": [],
        "subject": "New City Website Launch - March 1",
        "body": """Department PIOs,

The new City website launches March 1st! Here's what you need to know:

WHAT'S CHANGING:
- Modern, mobile-responsive design
- Improved search functionality
- Online service integration (permits, payments, requests)
- Accessibility enhancements (WCAG 2.1 AA compliant)
- Translation in 5 languages

CONTENT MIGRATION:
All current content has been migrated. However, please review your department pages by February 20th to ensure accuracy.

TRAINING SESSIONS:
- New CMS overview: Feb 6, 10am
- Editing your department pages: Feb 8, 2pm
- Advanced features: Feb 13, 10am

Register via Outlook calendar invitation (sent separately).

AFTER LAUNCH:
- You'll have editing access to your department pages
- Changes require Communications approval before publishing
- Major updates should be coordinated with our team

OLD WEBSITE:
Redirects will be in place. Old URLs will forward to new locations.

Send content review notes to webteam@cityofexample.gov

Communications Department""",
        "has_attachment": True,
        "attachment_names": ["Website_Style_Guide.pdf"]
    },
    {
        "id": "email_337",
        "thread_id": None,
        "thread_position": None,
        "thread_length": None,
        "date": "2023-05-03T10:30:00",
        "from": "communications@cityofexample.gov",
        "to": ["all-staff@cityofexample.gov"],
        "cc": [],
        "subject": "Media Inquiries - Protocol Reminder",
        "body": """All Staff,

A reminder on handling media inquiries:

IF A REPORTER CONTACTS YOU:
1. Be polite and professional
2. Get their name, outlet, and deadline
3. Say: "I'm not authorized to speak to media. Let me connect you with our Communications team."
4. Forward their contact info to communications@cityofexample.gov immediately
5. Do NOT answer questions about City business

WHAT NOT TO DO:
- Don't say "no comment" (sounds guilty)
- Don't speculate or guess
- Don't share personal opinions on City matters
- Don't refer them to someone else (let Communications handle)

AUTHORIZED SPOKESPERSONS:
- City Manager (overall City matters)
- Department Directors (their department, with Communications coordination)
- Communications Director (routine inquiries)

SOCIAL MEDIA INQUIRIES:
Treat the same as traditional media. Do not respond directly - forward to Communications.

CRISIS SITUATIONS:
Communications will activate our crisis protocol. All media inquiries go through the designated spokesperson ONLY.

Questions about this policy? Contact the Communications Director at ext. 4040.

Communications Department""",
        "has_attachment": False,
        "attachment_names": []
    },
    {
        "id": "email_338",
        "thread_id": None,
        "thread_position": None,
        "thread_length": None,
        "date": "2023-07-10T14:45:00",
        "from": "communications@cityofexample.gov",
        "to": ["department-heads@cityofexample.gov"],
        "cc": [],
        "subject": "City Newsletter - Content Submissions for August",
        "body": """Department Heads,

The August edition of the City Connections newsletter goes out August 1st. Submissions due July 21st.

THEME: Back to School / Emergency Preparedness Month

CONTENT REQUESTED:
- Department news and updates (100-150 words)
- Upcoming events for residents
- New services or programs
- Staff spotlights (new hires, promotions, retirements)
- Photos! (high-resolution, with captions and permission)

POPULAR TOPICS FOR AUGUST:
- School safety and traffic reminders
- Emergency kit preparation
- Back-to-school programs (Parks, Library)
- End-of-summer events

FORMAT:
- Submit in Word or plain text
- Include contact info for public inquiries
- Note any specific dates/deadlines to mention

DISTRIBUTION:
Newsletter goes to:
- 12,000 email subscribers
- City website
- Social media highlights
- Print copies at City facilities

Submit to newsletter@cityofexample.gov by July 21st.

Communications Department""",
        "has_attachment": False,
        "attachment_names": []
    },
    {
        "id": "email_339",
        "thread_id": None,
        "thread_position": None,
        "thread_length": None,
        "date": "2023-09-25T11:00:00",
        "from": "clerk@cityofexample.gov",
        "to": ["all-staff@cityofexample.gov"],
        "cc": [],
        "subject": "Public Meeting Reminder - Brown Act Compliance",
        "body": """All Staff,

With budget season approaching, a reminder about Brown Act compliance for public meetings:

THE BROWN ACT REQUIRES:
- 72-hour advance posting of agendas (regular meetings)
- 24-hour posting (special meetings)
- Meetings held in accessible locations
- Public comment opportunity
- No action on items not on the agenda

WHAT CONSTITUTES A "MEETING":
A majority of a legislative body (Council, Commission, Board) gathering to discuss, deliberate, or take action on matters within their jurisdiction.

This includes:
- Conference calls
- Serial communications that build consensus
- Social gatherings where business is discussed

STAFF ROLE:
- Work with City Clerk to post agendas on time
- Prepare materials well in advance
- Don't discuss agenda items with multiple members separately
- Direct public records requests to the Clerk

AGENDA POSTING:
- Submit items to City Clerk by Wednesday noon for the following Tuesday meeting
- Late items may be deferred to next meeting

PENALTIES FOR VIOLATIONS:
- Voided actions
- Misdemeanor charges
- Civil liability

Training available upon request. Contact the City Clerk's office.

City Clerk""",
        "has_attachment": True,
        "attachment_names": ["Brown_Act_Quick_Reference.pdf"]
    },
]


def get_reasoning(email):
    """Generate reasoning based on the email content."""
    subject = email["subject"].lower()

    reasoning_map = {
        "open enrollment": "Benefits/HR matter - employee insurance and health plans, unrelated to water or lead.",
        "performance review": "HR performance management - employee evaluation process, unrelated to water or lead.",
        "welcome new employee": "New employee announcement - routine HR onboarding, unrelated to water or lead.",
        "retirement": "Employee retirement celebration - personnel matter, unrelated to water or lead.",
        "harassment": "Mandatory compliance training - HR/legal requirement, unrelated to water or lead.",
        "holiday schedule": "Holiday schedule announcement - administrative matter, unrelated to water or lead.",
        "telework": "Telework policy - HR/administrative policy, unrelated to water or lead.",
        "assistance program": "Employee assistance program - benefits matter, unrelated to water or lead.",
        "ratification": "Union contract ratification - labor relations matter, unrelated to water or lead.",
        "wellness": "Employee wellness program - HR initiative, unrelated to water or lead.",
        "password": "IT security - password management, unrelated to water or lead.",
        "email system": "Email system maintenance - IT infrastructure, unrelated to water or lead.",
        "phishing": "Cybersecurity awareness - IT security training, unrelated to water or lead.",
        "laptop": "Computer equipment deployment - IT matter, unrelated to water or lead.",
        "teams": "Microsoft Teams training - IT/software matter, unrelated to water or lead.",
        "network": "Network issue resolution - IT infrastructure, unrelated to water or lead.",
        "gis": "GIS system launch - IT/mapping system, unrelated to water or lead.",
        "cybersecurity": "Cybersecurity training - IT security, unrelated to water or lead.",
        "parking": "Parking permit administration - facilities matter, unrelated to water or lead.",
        "retention": "Records retention policy - administrative/legal matter, unrelated to water or lead.",
        "office supply": "Office supply ordering - administrative procurement, unrelated to water or lead.",
        "casual": "Dress code policy - HR/administrative matter, unrelated to water or lead.",
        "badge": "Building access/security - facilities matter, unrelated to water or lead.",
        "vehicle": "City vehicle policy - fleet management, unrelated to water or lead.",
        "mail service": "Mail services - administrative operations, unrelated to water or lead.",
        "restroom": "Facility renovation - building maintenance, unrelated to water or lead.",
        "payroll": "Payroll administration - finance/HR matter, unrelated to water or lead.",
        "w-2": "Payroll administration - finance/HR matter, unrelated to water or lead.",
        "budget development": "Budget process - general finance matter, unrelated to water or lead.",
        "p-card": "Purchasing card policy - finance/procurement, unrelated to water or lead.",
        "mileage": "Expense reimbursement - finance policy, unrelated to water or lead.",
        "year-end": "Fiscal year-end procedures - accounting matter, unrelated to water or lead.",
        "budget approved": "Budget adoption announcement - general finance, unrelated to water or lead.",
        "audit": "Annual financial audit - accounting/compliance, unrelated to water or lead.",
        "petty cash": "Petty cash procedures - finance matter, unrelated to water or lead.",
        "picnic": "Employee social event - recreation/morale, unrelated to water or lead.",
        "blood drive": "Blood drive announcement - community health event, unrelated to water or lead.",
        "united way": "Charitable campaign - community/employee giving, unrelated to water or lead.",
        "food drive": "Food drive - charitable event, unrelated to water or lead.",
        "thanksgiving": "Food drive - charitable event, unrelated to water or lead.",
        "holiday party": "Holiday celebration - employee event, unrelated to water or lead.",
        "softball": "Employee recreation - sports team, unrelated to water or lead.",
        "hvac": "HVAC maintenance - building systems, unrelated to water or lead.",
        "fire drill": "Fire safety drill - emergency preparedness, unrelated to water or lead.",
        "elevator": "Elevator maintenance - building equipment, unrelated to water or lead.",
        "pest": "Pest control - facility maintenance, unrelated to water or lead.",
        "cleaning": "Cleaning services - janitorial matter, unrelated to water or lead.",
        "shakeout": "Earthquake drill - emergency preparedness, unrelated to water or lead.",
        "earthquake": "Earthquake drill - emergency preparedness, unrelated to water or lead.",
        "injury": "Workers compensation - risk management, unrelated to water or lead.",
        "workers": "Workers compensation - risk management, unrelated to water or lead.",
        "signature authority": "Contract authority - legal/administrative, unrelated to water or lead.",
        "insurance renewal": "Insurance renewal - risk management, unrelated to water or lead.",
        "social media": "Social media policy - communications/HR, unrelated to water or lead.",
        "gift": "Ethics policy - gift restrictions, unrelated to water or lead.",
        "ethics": "Ethics policy - gift restrictions, unrelated to water or lead.",
        "website": "Website launch - communications/IT, unrelated to water or lead.",
        "media inquir": "Media relations protocol - communications, unrelated to water or lead.",
        "newsletter": "Newsletter content request - communications, unrelated to water or lead.",
        "brown act": "Open meeting law compliance - legal/governance, unrelated to water or lead.",
        "public meeting": "Open meeting law compliance - legal/governance, unrelated to water or lead.",
    }

    for key, value in reasoning_map.items():
        if key in subject:
            return value

    return "General administrative matter - clearly unrelated to water, lead, or contamination."


def main():
    # Load existing emails
    with open("corpus/primary/emails.json", "r") as f:
        data = json.load(f)

    # Add new emails
    data["emails"].extend(new_emails)
    data["metadata"]["total_count"] = len(data["emails"])

    # Save updated emails
    with open("corpus/primary/emails.json", "w") as f:
        json.dump(data, f, indent=2)

    print(f"Added {len(new_emails)} TRUE_NEGATIVE emails")
    print(f"Total emails now: {data['metadata']['total_count']}")

    # Generate ground truth
    new_ground_truth = {}
    for email in new_emails:
        new_ground_truth[email["id"]] = {
            "responsive": False,
            "challenge_type": "TRUE_NEGATIVE",
            "buried_in_thread": False,
            "reasoning": get_reasoning(email),
            "keywords_present": [],
            "keywords_absent": ["lead", "contamination", "testing"]
        }

    # Load existing ground truth
    with open("corpus/primary/ground_truth.json", "r") as f:
        gt_data = json.load(f)

    # Add new ground truth entries
    gt_data["labels"].update(new_ground_truth)
    gt_data["metadata"]["total_non_responsive"] = sum(
        1 for v in gt_data["labels"].values() if not v["responsive"]
    )
    gt_data["metadata"]["by_challenge_type"]["TRUE_NEGATIVE"] = len(new_emails)

    # Save updated ground truth
    with open("corpus/primary/ground_truth.json", "w") as f:
        json.dump(gt_data, f, indent=2)

    print(f"Added {len(new_ground_truth)} ground truth entries")
    print(f"TRUE_NEGATIVE count: {len(new_emails)}")


if __name__ == "__main__":
    main()
