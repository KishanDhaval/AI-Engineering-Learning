"""
Sample data for the Product Catalog and Employee Directory.
In real-world applications, this would be backed by PostgreSQL, MongoDB,
an ERP system (SAP/Salesforce), or an internal HR database (Workday/BambooHR).
"""

from typing import Any, Dict, List, Optional

# ==========================================
# 1. PRODUCT CATALOG DATA
# ==========================================

PRODUCTS: List[Dict[str, Any]] = [
    {
        "id": "PROD-001",
        "name": "Quantum Pro Wireless Headphones",
        "category": "Electronics",
        "price": 149.99,
        "stock": 42,
        "rating": 4.7,
        "description": "Active noise cancelling over-ear headphones with 40-hour battery life and multi-point Bluetooth.",
        "specs": {
            "battery_life": "40 hours",
            "connectivity": "Bluetooth 5.3, 3.5mm jack",
            "weight": "250g",
            "color": "Midnight Black"
        }
    },
    {
        "id": "PROD-002",
        "name": "UltraVision 4K OLED Monitor 32-inch",
        "category": "Electronics",
        "price": 699.00,
        "stock": 15,
        "rating": 4.9,
        "description": "Professional color-accurate 4K OLED display with 144Hz refresh rate and 99% DCI-P3 gamut.",
        "specs": {
            "resolution": "3840 x 2160",
            "refresh_rate": "144Hz",
            "panel": "OLED",
            "ports": "Thunderbolt 4, HDMI 2.1, DisplayPort 1.4"
        }
    },
    {
        "id": "PROD-003",
        "name": "ErgoMotion Adjustable Standing Desk",
        "category": "Office Furniture",
        "price": 450.00,
        "stock": 0,  # Out of stock
        "rating": 4.5,
        "description": "Dual-motor electric standing desk with memory presets, solid bamboo desktop, and cable management tray.",
        "specs": {
            "dimensions": "60 x 30 inches",
            "height_range": "25 - 50 inches",
            "weight_capacity": "300 lbs",
            "frame_color": "Matte White"
        }
    },
    {
        "id": "PROD-004",
        "name": "Apex Mechanical Keyboard TKL",
        "category": "Electronics",
        "price": 89.50,
        "stock": 78,
        "rating": 4.6,
        "description": "Tenkeyless mechanical keyboard with hot-swappable linear switches and per-key RGB backlighting.",
        "specs": {
            "layout": "Tenkeyless (87 keys)",
            "switches": "Linear Red (Hot-swappable)",
            "keycaps": "Double-shot PBT",
            "connection": "Detachable USB-C"
        }
    },
    {
        "id": "PROD-005",
        "name": "CloudMesh Ergonomic Office Chair",
        "category": "Office Furniture",
        "price": 320.00,
        "stock": 23,
        "rating": 4.8,
        "description": "High-back breathable mesh ergonomic task chair with 4D armrests and dynamic lumbar support.",
        "specs": {
            "material": "Breathable mesh + aluminum base",
            "adjustability": "4D armrests, tilt tension, lumbar depth",
            "weight_limit": "330 lbs"
        }
    },
    {
        "id": "PROD-006",
        "name": "NanoBrew Precision Coffee Grinder",
        "category": "Kitchen & Appliances",
        "price": 125.00,
        "stock": 19,
        "rating": 4.4,
        "description": "Conical burr coffee grinder with 40 micro-adjustment grind settings and anti-static chute.",
        "specs": {
            "burr_type": "40mm Stainless Steel Conical",
            "hopper_capacity": "250g",
            "power": "150W"
        }
    }
]

# ==========================================
# 2. EMPLOYEE DIRECTORY DATA
# ==========================================

EMPLOYEES: List[Dict[str, Any]] = [
    {
        "id": "EMP-101",
        "name": "Aarav Sharma",
        "role": "Staff AI Engineer",
        "department": "Engineering",
        "email": "aarav.sharma@corp.example.com",
        "skills": ["Python", "MCP", "LangChain", "LLM Evaluation", "PyTorch"],
        "location": "Bengaluru, India",
        "status": "Active",
        "manager_id": "EMP-104",
        "bio": "Leads the AI Infrastructure team, specializing in MCP server architectures and autonomous agent orchestration."
    },
    {
        "id": "EMP-102",
        "name": "Elena Rostova",
        "role": "Senior Product Manager",
        "department": "Product",
        "email": "elena.rostova@corp.example.com",
        "skills": ["Product Strategy", "E-commerce Platforms", "Roadmapping", "Customer Analytics"],
        "location": "Berlin, Germany",
        "status": "Active",
        "manager_id": "EMP-105",
        "bio": "Manages core catalog experience and enterprise search capabilities."
    },
    {
        "id": "EMP-103",
        "name": "Marcus Chen",
        "role": "DevOps & Platform Architect",
        "department": "Infrastructure",
        "email": "marcus.chen@corp.example.com",
        "skills": ["Kubernetes", "FastAPI", "Docker", "GCP", "CI/CD"],
        "location": "San Francisco, USA",
        "status": "Active",
        "manager_id": "EMP-104",
        "bio": "Specializes in secure multi-cloud containerization and microservice gateway routing."
    },
    {
        "id": "EMP-104",
        "name": "Priya Nair",
        "role": "Director of Engineering",
        "department": "Engineering",
        "email": "priya.nair@corp.example.com",
        "skills": ["Engineering Leadership", "Distributed Systems", "Agentic Systems", "Cloud Governance"],
        "location": "Bengaluru, India",
        "status": "Active",
        "manager_id": None,
        "bio": "Oversees the Engineering, AI Platform, and DevOps organizations."
    },
    {
        "id": "EMP-105",
        "name": "David Miller",
        "role": "Head of Customer Support & Success",
        "department": "Customer Operations",
        "email": "david.miller@corp.example.com",
        "skills": ["Customer Success", "Escalation Management", "Zendesk", "SLA Monitoring"],
        "location": "Austin, USA",
        "status": "Active",
        "manager_id": None,
        "bio": "Directs customer onboarding, product warranty resolution, and enterprise support."
    }
]
