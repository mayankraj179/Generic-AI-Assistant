-- Sample table for the kb_demo_assistant database knowledge source.
-- Loaded into the kb_demo_source database by scripts/seed_kb_demo.py.
CREATE TABLE IF NOT EXISTS office_directory (
    office_code            text PRIMARY KEY,
    city                   text NOT NULL,
    address                text NOT NULL,
    it_helpdesk_extension  text NOT NULL,
    reception_hours        text NOT NULL,
    is_active              boolean NOT NULL DEFAULT true
);

INSERT INTO office_directory VALUES
    ('PUN', 'Pune', 'Tower B, 5th floor, Hinjewadi Phase 1', '4410', '08:30-19:00 Mon-Fri', true),
    ('MUM', 'Mumbai', 'Unit 1203, Powai Business Park', '4420', '09:00-18:30 Mon-Fri', true),
    ('BLR', 'Bengaluru', 'Block C, 3rd floor, Outer Ring Road', '4430', '09:00-19:00 Mon-Fri', true),
    ('NYC', 'New York', '14th floor, 250 Park Avenue South', '7710', '08:00-17:00 Mon-Fri', true),
    ('LDN', 'London', '2nd floor, 30 Old Street', '8810', '08:30-17:30 Mon-Fri', false)
ON CONFLICT (office_code) DO NOTHING;
