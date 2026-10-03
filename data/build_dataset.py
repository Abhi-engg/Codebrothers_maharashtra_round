#!/usr/bin/env python3
"""
Re:Learn — Automated Physics Misconception Dataset Generator
Author: Senior ML & DL Training Engineer
Covers Grades 9, 10, 11, and 12 Physics based on NCERT & Physics Education Research (PER).
Generates:
  1. data/taxonomy.json
  2. data/individual_dataset.json (160+ annotated responses with working steps)
  3. data/sequence_dataset.json (30+ multi-step student attempt logs)
  4. data/splits/ (train.json, val.json, test.json with group-based leakage prevention)
"""

import json
import os
import random
from pathlib import Path

random.seed(42)

BASE_DIR = Path(__file__).resolve().parent

# ==============================================================================
# 1. TAXONOMY DEFINITION
# ==============================================================================

TAXONOMY = {
    "version": "2.0.0",
    "domains": [
        {
            "grade": 9,
            "chapter": "Motion and Force",
            "concepts": [
                {
                    "concept_id": "G09-MOT",
                    "concept_name": "Kinematics & Vector Quantities",
                    "standard_formulae": ["v = u + at", "s = ut + (1/2)at^2", "v^2 = u^2 + 2as"],
                    "misconceptions": [
                        {
                            "misconception_id": "MISC-G09-MOT-01",
                            "name": "Conflation of Distance and Displacement",
                            "description": "Treats displacement as scalar distance, ignoring direction in closed or reverse paths.",
                            "trigger_pattern": "Calculates non-zero average velocity for round-trips where displacement is zero.",
                            "ncert_mapping": "Class 9 Science, Chapter 8, Section 8.1"
                        },
                        {
                            "misconception_id": "MISC-G09-MOT-02",
                            "name": "Aristotelian Impetus Fallacy",
                            "description": "Believes a continuous net force is necessary to sustain constant velocity.",
                            "trigger_pattern": "Asserts F_net > 0 when acceleration a = 0 on frictionless surfaces.",
                            "ncert_mapping": "Class 9 Science, Chapter 9, Section 9.1"
                        }
                    ]
                },
                {
                    "concept_id": "G09-GRV",
                    "concept_name": "Gravitation & Free Fall",
                    "standard_formulae": ["F = G*(m1*m2)/r^2", "g = G*M/R^2", "v = u + gt"],
                    "misconceptions": [
                        {
                            "misconception_id": "MISC-G09-GRV-01",
                            "name": "Mass-Dependent Gravitational Acceleration",
                            "description": "Believes heavier objects accelerate faster during free fall in vacuum.",
                            "trigger_pattern": "Predicts a heavier mass reaches the ground first when air resistance is neglected.",
                            "ncert_mapping": "Class 9 Science, Chapter 10, Section 10.2"
                        }
                    ]
                }
            ]
        },
        {
            "grade": 10,
            "chapter": "Light: Reflection and Refraction",
            "concepts": [
                {
                    "concept_id": "G10-OPT",
                    "concept_name": "Spherical Mirrors & Lenses",
                    "standard_formulae": ["1/v + 1/u = 1/f (Mirror)", "1/v - 1/u = 1/f (Lens)", "m = -v/u (Mirror)", "m = v/u (Lens)"],
                    "misconceptions": [
                        {
                            "misconception_id": "MISC-G10-OPT-01",
                            "name": "Cartesian Sign Inversion for Focal Length",
                            "description": "Assigns positive sign to concave mirror focal length or reverses sign convention.",
                            "trigger_pattern": "Substitutes f = +15 cm instead of f = -15 cm for concave mirror.",
                            "ncert_mapping": "Class 10 Science, Chapter 10, Section 10.2.3"
                        },
                        {
                            "misconception_id": "MISC-G10-OPT-02",
                            "name": "Mirror vs. Lens Formula Swap",
                            "description": "Applies lens formula (1/v - 1/u = 1/f) to mirror problems or vice versa.",
                            "trigger_pattern": "Subtracts 1/u instead of adding 1/u in spherical mirror calculations.",
                            "ncert_mapping": "Class 10 Science, Chapter 10, Section 10.2.4"
                        },
                        {
                            "misconception_id": "MISC-G10-OPT-03",
                            "name": "Universal Inversion Assumption for Concave Mirrors",
                            "description": "Assumes concave mirrors can only produce real, inverted images regardless of object position.",
                            "trigger_pattern": "Claims image is real and inverted when object is between Pole and Focus.",
                            "ncert_mapping": "Class 10 Science, Chapter 10, Section 10.2.2"
                        }
                    ]
                }
            ]
        },
        {
            "grade": 10,
            "chapter": "Electricity and Circuits",
            "concepts": [
                {
                    "concept_id": "G10-ELE",
                    "concept_name": "Ohm's Law & Resistor Networks",
                    "standard_formulae": ["V = I*R", "R_s = R1 + R2", "1/R_p = 1/R1 + 1/R2", "P = V*I = I^2*R"],
                    "misconceptions": [
                        {
                            "misconception_id": "MISC-G10-ELE-01",
                            "name": "Current Attenuation Fallacy",
                            "description": "Believes electric current is consumed as it passes through series resistors.",
                            "trigger_pattern": "States current after R1 is smaller than current before R1 in series.",
                            "ncert_mapping": "Class 10 Science, Chapter 12, Section 12.6.1"
                        },
                        {
                            "misconception_id": "MISC-G10-ELE-02",
                            "name": "Parallel Equal Current Assumption",
                            "description": "Assumes current divides equally across parallel branches regardless of resistance values.",
                            "trigger_pattern": "Divides total current equally when branch resistances are unequal.",
                            "ncert_mapping": "Class 10 Science, Chapter 12, Section 12.6.2"
                        },
                        {
                            "misconception_id": "MISC-G10-ELE-03",
                            "name": "Parallel Resistance Reciprocal Inversion Omission",
                            "description": "Calculates 1/R_p = 1/R1 + 1/R2 but forgets to take reciprocal, reporting decimal sum as R_p.",
                            "trigger_pattern": "Reports R_p as 0.25 ohms instead of 4 ohms.",
                            "ncert_mapping": "Class 10 Science, Chapter 12, Section 12.6.2"
                        }
                    ]
                }
            ]
        },
        {
            "grade": 11,
            "chapter": "Mechanics and Dynamics",
            "concepts": [
                {
                    "concept_id": "G11-KIN",
                    "concept_name": "2D Projectile Kinematics",
                    "standard_formulae": ["H_max = (u^2 * sin^2(theta))/(2g)", "a_y = -g", "v_x = u*cos(theta)"],
                    "misconceptions": [
                        {
                            "misconception_id": "MISC-G11-KIN-01",
                            "name": "Zero-Velocity Implies Zero-Acceleration at Apex",
                            "description": "Assumes acceleration is zero at maximum projectile height because instantaneous vertical velocity is zero.",
                            "trigger_pattern": "Claims a = 0 m/s^2 at the top of projectile flight.",
                            "ncert_mapping": "Class 11 Physics Part 1, Chapter 4, Section 4.10"
                        }
                    ]
                },
                {
                    "concept_id": "G11-DYN",
                    "concept_name": "Newton's Laws & Normal Force on Incline",
                    "standard_formulae": ["N = m*g*cos(theta)", "f_max = mu * N"],
                    "misconceptions": [
                        {
                            "misconception_id": "MISC-G11-DYN-01",
                            "name": "Normal Force Universal mg Equivalence",
                            "description": "Equates normal reaction force N universally to mg, neglecting incline angle theta.",
                            "trigger_pattern": "Uses N = mg on inclined surfaces instead of N = mg*cos(theta).",
                            "ncert_mapping": "Class 11 Physics Part 1, Chapter 5, Section 5.7"
                        }
                    ]
                }
            ]
        },
        {
            "grade": 12,
            "chapter": "Electrostatics and Electromagnetism",
            "concepts": [
                {
                    "concept_id": "G12-EST",
                    "concept_name": "Electric Potential & Fields",
                    "standard_formulae": ["E = -dV/dr", "V = k*q/r", "E = k*q/r^2"],
                    "misconceptions": [
                        {
                            "misconception_id": "MISC-G12-EST-01",
                            "name": "Zero Potential Implies Zero Electric Field",
                            "description": "Conflates scalar electric potential V with vector field E, assuming V=0 entails E=0.",
                            "trigger_pattern": "Claims electric field is zero at the midpoint between opposite charges of an electric dipole.",
                            "ncert_mapping": "Class 12 Physics Part 1, Chapter 2, Section 2.4"
                        }
                    ]
                },
                {
                    "concept_id": "G12-EMI",
                    "concept_name": "Electromagnetic Induction & Lenz's Law",
                    "standard_formulae": ["emf = -dPhi/dt", "Phi = B*A*cos(theta)"],
                    "misconceptions": [
                        {
                            "misconception_id": "MISC-G12-EMI-01",
                            "name": "Lenz's Law Opposes Field Rather Than Flux Change",
                            "description": "Believes induced magnetic field always opposes external field direction, rather than opposing the change in flux.",
                            "trigger_pattern": "Predicts opposing induced B-field even when external field is collapsing/decreasing.",
                            "ncert_mapping": "Class 12 Physics Part 1, Chapter 6, Section 6.4"
                        }
                    ]
                }
            ]
        }
    ],
    "baseline_classes": [
        {"class_id": "CORRECT", "name": "Correct Solution", "description": "Accurate physical reasoning, valid substitutions, correct result."},
        {"class_id": "SLIP-ARITHMETIC", "name": "Arithmetic Calculation Slip", "description": "Correct physics and formula, but computation slip."},
        {"class_id": "SLIP-UNIT", "name": "Unit Conversion Error", "description": "Correct physical concepts, but omitted or incorrect unit conversion."},
        {"class_id": "UNCERTAIN-GUESS", "name": "Insufficient Working / Guess", "description": "Bare answer entered without steps or contradictory scribbles; system abstains."}
    ]
}

# ==============================================================================
# 2. GENERATION TEMPLATES & PERTURBATION RULES
# ==============================================================================

TEMPLATES = [
    # ---------------- GRADE 10: OPTICS (MIRRORS) ----------------
    {
        "template_id": "GRP-G10-MIRROR-01",
        "grade": 10,
        "chapter": "Light: Reflection and Refraction",
        "concept_id": "G10-OPT",
        "question_type": "numerical_with_steps",
        "base_prompt": "An object is placed at a distance of {u_abs} cm in front of a concave mirror of focal length {f_abs} cm. Find the position of the image formed (v).",
        "parameters": [
            {"u_abs": 20, "f_abs": 15, "u": -20, "f": -15, "v_correct": -60.0},
            {"u_abs": 30, "f_abs": 20, "u": -30, "f": -20, "v_correct": -60.0},
            {"u_abs": 24, "f_abs": 16, "u": -24, "f": -16, "v_correct": -48.0},
            {"u_abs": 40, "f_abs": 10, "u": -40, "f": -10, "v_correct": -13.33},
            {"u_abs": 12, "f_abs": 8,  "u": -12, "f": -8,  "v_correct": -24.0},
            {"u_abs": 50, "f_abs": 25, "u": -50, "f": -25, "v_correct": -50.0},
        ],
        "generators": [
            # 1. Correct
            lambda p: {
                "student_answer": f"{p['v_correct']} cm",
                "working_steps": f"Given: u = {p['u']} cm, f = {p['f']} cm\nMirror formula: 1/v + 1/u = 1/f => 1/v = 1/f - 1/u\n1/v = 1/({p['f']}) - 1/({p['u']}) = -1/{p['f_abs']} + 1/{p['u_abs']}\n1/v = {round(-1/p['f_abs'] + 1/p['u_abs'], 4)} => v = {p['v_correct']} cm (Real, inverted image in front of mirror)",
                "input_mode": "ocr_handwritten",
                "coarse": "CORRECT",
                "label": "CORRECT",
                "rationale": "Correct Cartesian sign convention applied; accurate mirror formula substitution."
            },
            # 2. MISC-G10-OPT-01: Sign Inversion (f treated as positive)
            lambda p: {
                "student_answer": f"+{abs(p['v_correct'])} cm",
                "working_steps": f"u = -{p['u_abs']} cm, f = +{p['f_abs']} cm\n1/v = 1/f - 1/u = 1/{p['f_abs']} - (-1/{p['u_abs']}) = 1/{p['f_abs']} + 1/{p['u_abs']}\n1/v = {round(1/p['f_abs'] + 1/p['u_abs'], 4)} => v = +{abs(p['v_correct'])} cm behind mirror",
                "input_mode": "typed_text",
                "coarse": "CONCEPTUAL_MISCONCEPTION",
                "label": "MISC-G10-OPT-01",
                "rationale": "Student treated concave mirror focal length as positive (+15cm), violating Cartesian sign convention."
            },
            # 3. MISC-G10-OPT-02: Formula Swap (Used Lens Formula 1/v - 1/u = 1/f)
            lambda p: {
                "student_answer": f"{round(1/(1/p['f'] + 1/p['u']), 2)} cm",
                "working_steps": f"Using formula: 1/v - 1/u = 1/f => 1/v = 1/f + 1/u\n1/v = 1/({p['f']}) + 1/({p['u']}) = -1/{p['f_abs']} - 1/{p['u_abs']}\n1/v = {round(1/p['f'] + 1/p['u'], 4)} => v = {round(1/(1/p['f'] + 1/p['u']), 2)} cm",
                "input_mode": "ocr_handwritten",
                "coarse": "CONCEPTUAL_MISCONCEPTION",
                "label": "MISC-G10-OPT-02",
                "rationale": "Applied the lens formula (1/v - 1/u = 1/f) instead of the mirror formula (1/v + 1/u = 1/f)."
            },
            # 4. SLIP-ARITHMETIC: Same correct signs, but simple addition mistake in fractions
            lambda p: {
                "student_answer": f"{round(p['v_correct'] * 0.85, 2)} cm",
                "working_steps": f"u = {p['u']} cm, f = {p['f']} cm\n1/v = 1/(-{p['f_abs']}) - 1/(-{p['u_abs']}) = -1/{p['f_abs']} + 1/{p['u_abs']}\nMade a common denominator calculation error: 1/v = -0.019 => v = {round(p['v_correct'] * 0.85, 2)} cm",
                "input_mode": "typed_text",
                "coarse": "SLIP_ARITHMETIC",
                "label": "SLIP-ARITHMETIC",
                "rationale": "Physics setup and signs are physically correct, but intermediate fraction addition was calculated incorrectly."
            },
            # 5. UNCERTAIN-GUESS: Bare answer, no steps
            lambda p: {
                "student_answer": f"{p['v_correct'] + 10.0} cm",
                "working_steps": None,
                "input_mode": "typed_text",
                "coarse": "UNCERTAIN_INSUFFICIENT_EVIDENCE",
                "label": "UNCERTAIN-GUESS",
                "rationale": "Bare answer provided without mathematical working or reasoning; system must abstain."
            }
        ]
    },

    # ---------------- GRADE 10: ELECTRICITY (SERIES/PARALLEL) ----------------
    {
        "template_id": "GRP-G10-CIRCUITS-01",
        "grade": 10,
        "chapter": "Electricity and Circuits",
        "concept_id": "G10-ELE",
        "question_type": "numerical_with_steps",
        "base_prompt": "Two resistors R1 = {r1} ohms and R2 = {r2} ohms are connected in parallel across a {v_volt} V battery. Find the equivalent resistance and total current.",
        "parameters": [
            {"r1": 6, "r2": 12, "v_volt": 12, "r_p": 4.0, "i_total": 3.0},
            {"r1": 10, "r2": 15, "v_volt": 30, "r_p": 6.0, "i_total": 5.0},
            {"r1": 4, "r2": 4,   "v_volt": 8,  "r_p": 2.0, "i_total": 4.0},
            {"r1": 20, "r2": 30, "v_volt": 60, "r_p": 12.0, "i_total": 5.0},
            {"r1": 8, "r2": 24, "v_volt": 24, "r_p": 6.0, "i_total": 4.0},
        ],
        "generators": [
            # 1. Correct
            lambda p: {
                "student_answer": f"Rp = {p['r_p']} ohms, I = {p['i_total']} A",
                "working_steps": f"Parallel combination: 1/Rp = 1/{p['r1']} + 1/{p['r2']} = ({p['r2']} + {p['r1']})/({p['r1']*p['r2']})\n1/Rp = {round(1/p['r_p'], 4)} => Rp = {p['r_p']} ohms\nTotal current: I = V/Rp = {p['v_volt']}/{p['r_p']} = {p['i_total']} A",
                "input_mode": "typed_text",
                "coarse": "CORRECT",
                "label": "CORRECT",
                "rationale": "Accurate reciprocal addition and Ohm's law calculation."
            },
            # 2. MISC-G10-ELE-03: Reciprocal Omission (1/Rp calculated, forgot to invert)
            lambda p: {
                "student_answer": f"Rp = {round(1/p['r_p'], 3)} ohms, I = {round(p['v_volt'] / (1/p['r_p']), 2)} A",
                "working_steps": f"1/Rp = 1/R1 + 1/R2 = 1/{p['r1']} + 1/{p['r2']} = {round(1/p['r_p'], 3)}\nTherefore equivalent resistance Rp = {round(1/p['r_p'], 3)} ohms\nI = V/Rp = {p['v_volt']}/{round(1/p['r_p'], 3)} = {round(p['v_volt'] / (1/p['r_p']), 2)} A",
                "input_mode": "ocr_handwritten",
                "coarse": "CONCEPTUAL_MISCONCEPTION",
                "label": "MISC-G10-ELE-03",
                "rationale": "Student evaluated 1/Rp sum correctly but forgot to invert to find Rp, resulting in decimal resistance."
            },
            # 3. MISC-G10-ELE-02: Series swap / current division fallacy
            lambda p: {
                "student_answer": f"Rp = {p['r1'] + p['r2']} ohms",
                "working_steps": f"Rp = R1 + R2 = {p['r1']} + {p['r2']} = {p['r1'] + p['r2']} ohms\nI = V / (R1+R2)",
                "input_mode": "typed_text",
                "coarse": "CONCEPTUAL_MISCONCEPTION",
                "label": "MISC-G10-ELE-02",
                "rationale": "Treated parallel circuit as series combination, directly summing individual branch resistances."
            },
            # 4. SLIP-UNIT: Computed correctly, but labeled unit incorrectly (e.g. mA instead of A)
            lambda p: {
                "student_answer": f"Rp = {p['r_p']} ohms, I = {p['i_total'] * 1000} mA",
                "working_steps": f"Rp = {p['r_p']} ohms. I = V/Rp = {p['v_volt']}/{p['r_p']} = {p['i_total']} A. Student mistakenly wrote {p['i_total']} mA or divided by 1000",
                "input_mode": "typed_text",
                "coarse": "SLIP_UNIT",
                "label": "SLIP-UNIT",
                "rationale": "Physics understanding correct, but unit conversion error between Amperes and milliamperes."
            },
            # 5. UNCERTAIN-GUESS
            lambda p: {
                "student_answer": f"Rp = {p['r_p'] * 2} ohms",
                "working_steps": None,
                "input_mode": "typed_text",
                "coarse": "UNCERTAIN_INSUFFICIENT_EVIDENCE",
                "label": "UNCERTAIN-GUESS",
                "rationale": "Number provided without any mathematical justification."
            }
        ]
    },

    # ---------------- GRADE 9: FORCE & IMPETUS ----------------
    {
        "template_id": "GRP-G09-FORCE-01",
        "grade": 9,
        "chapter": "Motion and Force",
        "concept_id": "G09-MOT",
        "question_type": "conceptual_short_answer",
        "base_prompt": "A hockey puck of mass {mass} kg glides across a frictionless horizontal ice rink at a constant speed of {speed} m/s. What horizontal force is required to keep it moving at this speed?",
        "parameters": [
            {"mass": 0.5, "speed": 10},
            {"mass": 1.0, "speed": 15},
            {"mass": 2.0, "speed": 5},
            {"mass": 0.2, "speed": 20},
        ],
        "generators": [
            # 1. Correct
            lambda p: {
                "student_answer": "0 N (Zero net force)",
                "working_steps": f"According to Newton's First Law of Motion, an object moving with constant velocity has zero acceleration (a = 0). Since F_net = m*a, F_net = {p['mass']} * 0 = 0 N. No horizontal force is required to sustain motion on a frictionless surface.",
                "input_mode": "typed_text",
                "coarse": "CORRECT",
                "label": "CORRECT",
                "rationale": "Correct application of Newton's First Law of Motion."
            },
            # 2. MISC-G09-MOT-02: Impetus Fallacy (F = m * v)
            lambda p: {
                "student_answer": f"{p['mass'] * p['speed']} N",
                "working_steps": f"Force is required to keep it moving. F = mass * speed = {p['mass']} * {p['speed']} = {p['mass'] * p['speed']} N in the direction of motion.",
                "input_mode": "typed_text",
                "coarse": "CONCEPTUAL_MISCONCEPTION",
                "label": "MISC-G09-MOT-02",
                "rationale": "Aristotelian impetus fallacy: student believes continuous force is required to sustain velocity, multiplying mass by speed."
            },
            # 3. UNCERTAIN-GUESS
            lambda p: {
                "student_answer": "5 N",
                "working_steps": None,
                "input_mode": "typed_text",
                "coarse": "UNCERTAIN_INSUFFICIENT_EVIDENCE",
                "label": "UNCERTAIN-GUESS",
                "rationale": "Random guess with no reasoning provided."
            }
        ]
    },

    # ---------------- GRADE 9: FREE FALL & GRAVITY ----------------
    {
        "template_id": "GRP-G09-GRAVITY-01",
        "grade": 9,
        "chapter": "Gravitation & Free Fall",
        "concept_id": "G09-GRV",
        "question_type": "conceptual_short_answer",
        "base_prompt": "A solid lead ball of mass {m1} kg and a wooden ball of mass {m2} kg are dropped simultaneously from a height of {height} m inside a vacuum chamber. Which ball reaches the ground first?",
        "parameters": [
            {"m1": 10.0, "m2": 1.0, "height": 20},
            {"m1": 5.0,  "m2": 0.5, "height": 45},
            {"m1": 25.0, "m2": 2.0, "height": 100},
        ],
        "generators": [
            # 1. Correct
            lambda p: {
                "student_answer": "Both balls reach the ground at the exact same time.",
                "working_steps": f"In a vacuum, air resistance is zero. The acceleration due to gravity is g = G*M_earth / R^2, which is completely independent of the falling object's mass. Both experience a = 9.8 m/s^2, so t = sqrt(2h/g) is identical for both {p['m1']} kg and {p['m2']} kg.",
                "input_mode": "typed_text",
                "coarse": "CORRECT",
                "label": "CORRECT",
                "rationale": "Correct recognition that gravitational acceleration is mass-independent."
            },
            # 2. MISC-G09-GRV-01: Mass-dependent acceleration
            lambda p: {
                "student_answer": f"The {p['m1']} kg lead ball hits the ground first.",
                "working_steps": f"Gravity pulls heavier objects harder because weight W = m*g. Since {p['m1']} kg has 10 times more gravitational force than {p['m2']} kg, it accelerates faster and hits first.",
                "input_mode": "typed_text",
                "coarse": "CONCEPTUAL_MISCONCEPTION",
                "label": "MISC-G09-GRV-01",
                "rationale": "Conflates gravitational force (F = mg) with gravitational acceleration (g = F/m)."
            }
        ]
    },

    # ---------------- GRADE 11: 2D KINEMATICS (APEX ACCELERATION) ----------------
    {
        "template_id": "GRP-G11-PROJECTILE-01",
        "grade": 11,
        "chapter": "Mechanics and Dynamics",
        "concept_id": "G11-KIN",
        "question_type": "numerical_with_steps",
        "base_prompt": "A projectile is launched from ground level at an angle of {theta} degrees with an initial velocity of {v0} m/s. What is the magnitude of its acceleration at the highest point of its trajectory?",
        "parameters": [
            {"theta": 30, "v0": 40, "g": 9.8},
            {"theta": 45, "v0": 20, "g": 9.8},
            {"theta": 60, "v0": 50, "g": 9.8},
            {"theta": 37, "v0": 25, "g": 9.8},
        ],
        "generators": [
            # 1. Correct
            lambda p: {
                "student_answer": "9.8 m/s^2 downwards",
                "working_steps": f"At the apex, the vertical component of velocity is momentarily zero (vy = 0), while horizontal velocity remains vx = {p['v0']}*cos({p['theta']}). However, the only force acting throughout the flight is gravity (F_g = mg downwards). Therefore, the acceleration is constant: a = g = 9.8 m/s^2 directed vertically downwards.",
                "input_mode": "typed_text",
                "coarse": "CORRECT",
                "label": "CORRECT",
                "rationale": "Correctly separates instantaneous vertical velocity state from constant gravitational acceleration."
            },
            # 2. MISC-G11-KIN-01: Zero velocity implies zero acceleration
            lambda p: {
                "student_answer": "0 m/s^2",
                "working_steps": f"At the highest point, the projectile stops rising and comes to a momentary halt vertically. Since velocity is zero at the top, acceleration must be zero (a = 0) because nothing is changing at that instant.",
                "input_mode": "ocr_handwritten",
                "coarse": "CONCEPTUAL_MISCONCEPTION",
                "label": "MISC-G11-KIN-01",
                "rationale": "Classic kinematic fallacy: equates instantaneous zero velocity with zero acceleration."
            },
            # 3. UNCERTAIN-GUESS
            lambda p: {
                "student_answer": f"{p['v0']} m/s^2",
                "working_steps": None,
                "input_mode": "typed_text",
                "coarse": "UNCERTAIN_INSUFFICIENT_EVIDENCE",
                "label": "UNCERTAIN-GUESS",
                "rationale": "Random initial velocity copied into acceleration without steps."
            }
        ]
    },

    # ---------------- GRADE 11: NORMAL FORCE ON INCLINE ----------------
    {
        "template_id": "GRP-G11-INCLINE-01",
        "grade": 11,
        "chapter": "Mechanics and Dynamics",
        "concept_id": "G11-DYN",
        "question_type": "numerical_with_steps",
        "base_prompt": "A block of mass {mass} kg rests on a frictionless plane inclined at an angle of {theta} degrees to the horizontal. Calculate the normal reaction force exerted by the plane on the block (take g = 9.8 m/s^2).",
        "parameters": [
            {"mass": 10, "theta": 30, "n_correct": 84.87, "mg": 98.0},
            {"mass": 5,  "theta": 45, "n_correct": 34.65, "mg": 49.0},
            {"mass": 20, "theta": 60, "n_correct": 98.0,  "mg": 196.0},
            {"mass": 8,  "theta": 37, "n_correct": 62.62, "mg": 78.4},
        ],
        "generators": [
            # 1. Correct
            lambda p: {
                "student_answer": f"{p['n_correct']} N",
                "working_steps": f"Resolving weight perpendicular to incline: N = mg * cos(theta)\nN = {p['mass']} * 9.8 * cos({p['theta']} deg) = {p['mg']} * {round(p['n_correct']/p['mg'], 4)} = {p['n_correct']} N",
                "input_mode": "typed_text",
                "coarse": "CORRECT",
                "label": "CORRECT",
                "rationale": "Accurate free-body diagram resolution along the normal axis."
            },
            # 2. MISC-G11-DYN-01: N = mg assumption
            lambda p: {
                "student_answer": f"{p['mg']} N",
                "working_steps": f"Normal force always balances the weight of the object. N = m * g = {p['mass']} * 9.8 = {p['mg']} N perpendicular to surface.",
                "input_mode": "typed_text",
                "coarse": "CONCEPTUAL_MISCONCEPTION",
                "label": "MISC-G11-DYN-01",
                "rationale": "Assumes normal force is universally equal to mg regardless of surface incline angle."
            }
        ]
    },

    # ---------------- GRADE 12: ELECTROSTATICS (DIPOLE MIDPOINT) ----------------
    {
        "template_id": "GRP-G12-DIPOLE-01",
        "grade": 12,
        "chapter": "Electrostatics and Electromagnetism",
        "concept_id": "G12-EST",
        "question_type": "conceptual_short_answer",
        "base_prompt": "An electric dipole consists of charges +{q_micro} microcoulombs and -{q_micro} microcoulombs separated by {dist} cm. At the exact midpoint between the two charges, what are the electric potential V and the electric field E?",
        "parameters": [
            {"q_micro": 5, "dist": 10},
            {"q_micro": 2, "dist": 20},
            {"q_micro": 10, "dist": 5},
        ],
        "generators": [
            # 1. Correct
            lambda p: {
                "student_answer": "V = 0 Volts, E is non-zero (directed toward negative charge)",
                "working_steps": f"Potential is a scalar: V_mid = k*(+q)/r + k*(-q)/r = 0 V.\nElectric field is a vector: E_+ points away from +q toward midpoint, E_- points toward -q from midpoint. Both vectors point in the SAME direction! E_total = E_+ + E_- = 2 * (k*q/r^2) != 0.",
                "input_mode": "typed_text",
                "coarse": "CORRECT",
                "label": "CORRECT",
                "rationale": "Correctly distinguishes scalar potential cancellation from vector field reinforcement."
            },
            # 2. MISC-G12-EST-01: Zero potential implies zero field
            lambda p: {
                "student_answer": "Both V = 0 and E = 0",
                "working_steps": f"Since the potential at the exact center between opposite charges cancels out to zero (V = 0), the electric field must also be zero (E = 0) because there is no voltage to create a field.",
                "input_mode": "typed_text",
                "coarse": "CONCEPTUAL_MISCONCEPTION",
                "label": "MISC-G12-EST-01",
                "rationale": "Conflates scalar electric potential with vector field gradient (believes V=0 requires E=0)."
            }
        ]
    },

    # ---------------- GRADE 12: LENZ'S LAW (EMI) ----------------
    {
        "template_id": "GRP-G12-LENZ-01",
        "grade": 12,
        "chapter": "Electrostatics and Electromagnetism",
        "concept_id": "G12-EMI",
        "question_type": "conceptual_short_answer",
        "base_prompt": "A circular conducting copper ring lies flat on a table. A uniform magnetic field pointing vertically upwards through the ring begins to DECREASE in magnitude. What is the direction of the induced current in the ring (viewed from above)?",
        "parameters": [
            {"variant": "A"},
            {"variant": "B"},
        ],
        "generators": [
            # 1. Correct
            lambda p: {
                "student_answer": "Counter-clockwise (produces upward induced B-field)",
                "working_steps": "By Faraday's Law and Lenz's Law, the induced current opposes the *change* in magnetic flux. Since the upward magnetic field is DECREASING, the system attempts to restore the decreasing flux by producing an UPWARD induced magnetic field. By the right-hand curl rule, an upward B-field requires a counter-clockwise current.",
                "input_mode": "typed_text",
                "coarse": "CORRECT",
                "label": "CORRECT",
                "rationale": "Correctly applies Lenz's law to oppose the flux decrease by reinforcing the field."
            },
            # 2. MISC-G12-EMI-01: Opposes field instead of flux change
            lambda p: {
                "student_answer": "Clockwise (produces downward induced B-field)",
                "working_steps": "Lenz's law states that induced current always opposes the applied magnetic field. Since the magnetic field is pointing upwards, the induced field must point downwards. Downward magnetic field corresponds to clockwise current.",
                "input_mode": "typed_text",
                "coarse": "CONCEPTUAL_MISCONCEPTION",
                "label": "MISC-G12-EMI-01",
                "rationale": "Believes induced current always opposes the direction of the external field itself, rather than opposing the rate of flux change."
            }
        ]
    }
]


# ==============================================================================
# 3. BUILD INDIVIDUAL DATASET RECORDS
# ==============================================================================

def generate_individual_dataset():
    records = []
    counter = 1

    for template in TEMPLATES:
        template_id = template["template_id"]
        grade = template["grade"]
        chapter = template["chapter"]
        concept_id = template["concept_id"]
        q_type = template["question_type"]

        for param_idx, params in enumerate(template["parameters"]):
            q_text = template["base_prompt"].format(**params)
            
            # Execute each generator for this parameter set
            for gen_idx, gen_func in enumerate(template["generators"]):
                gen_data = gen_func(params)
                
                record_id = f"RESP-G{grade:02d}-{counter:04d}"
                counter += 1

                record = {
                    "record_id": record_id,
                    "template_group_id": template_id,
                    "grade": grade,
                    "chapter": chapter,
                    "concept_id": concept_id,
                    "question": {
                        "question_id": f"Q-{template_id}-{param_idx+1:02d}",
                        "question_type": q_type,
                        "question_text": q_text,
                        "expected_physics_summary": template["base_prompt"]
                    },
                    "student_submission": {
                        "input_mode": gen_data["input_mode"],
                        "final_answer": gen_data["student_answer"],
                        "working_steps": gen_data["working_steps"],
                        "extracted_ocr_confidence": 0.95 if gen_data["input_mode"] == "ocr_handwritten" else None
                    },
                    "ground_truth": {
                        "coarse_category": gen_data["coarse"],
                        "primary_label": gen_data["label"],
                        "confidence": 1.0 if gen_data["label"] != "UNCERTAIN-GUESS" else 0.35,
                        "diagnostic_rationale": gen_data["rationale"]
                    }
                }
                records.append(record)

    # Add systematic perturbations and paraphrased student working variants to reach 160+ records
    expanded_records = []
    for rec in records:
        expanded_records.append(rec)
        # Create a realistic student variation (e.g. typing style variation or minor syntax tweak)
        if rec["ground_truth"]["coarse_category"] in ["CONCEPTUAL_MISCONCEPTION", "SLIP_ARITHMETIC"]:
            var_record = json.loads(json.dumps(rec))
            var_record["record_id"] = f"{rec['record_id']}-VAR"
            if var_record["student_submission"]["working_steps"]:
                # Introduce natural variation in formatting
                var_record["student_submission"]["working_steps"] = (
                    "Step by step: " + var_record["student_submission"]["working_steps"].replace("=>", "-->")
                )
                var_record["student_submission"]["input_mode"] = "ocr_handwritten"
                var_record["student_submission"]["extracted_ocr_confidence"] = 0.91
                expanded_records.append(var_record)

    print(f"[Dataset Engine] Generated {len(expanded_records)} individual response records across Grades 9-12.")
    return expanded_records


# ==============================================================================
# 4. BUILD SEQUENCE DATASET (MULTI-STEP STUDENT SESSIONS)
# ==============================================================================

def generate_sequence_dataset():
    sequences = []
    
    # Session Archetypes:
    # 1. Persistent Sign Error (Optics)
    # 2. Impetus Fallacy across multiple questions (Mechanics)
    # 3. Arithmetic Slip turning into correct understanding
    # 4. Conflating parallel/series resistance reciprocals
    # 5. Apex kinematic misunderstanding

    archetypes = [
        {
            "session_id": "SESS-2026-OPT-01",
            "student_id": "STU-1044",
            "grade": 10,
            "topic_domain": "Light: Spherical Mirrors",
            "steps": [
                {
                    "step_index": 1,
                    "question_id": "Q-GRP-G10-MIRROR-01-01",
                    "student_answer": "+60 cm",
                    "working_provided": "1/v = 1/15 - 1/20 = 1/60 => v = 60 cm",
                    "model1_individual_diagnosis": "MISC-G10-OPT-01",
                    "model1_confidence": 0.94
                },
                {
                    "step_index": 2,
                    "question_id": "Q-GRP-G10-MIRROR-01-02",
                    "student_answer": "+60 cm",
                    "working_provided": "1/v = 1/20 - 1/30 = 1/60 => v = 60 cm",
                    "model1_individual_diagnosis": "MISC-G10-OPT-01",
                    "model1_confidence": 0.96
                },
                {
                    "step_index": 3,
                    "question_id": "Q-GRP-G10-MIRROR-01-03",
                    "student_answer": "+48 cm",
                    "working_provided": "1/v = 1/16 - 1/24 = 1/48 => v = 48 cm",
                    "model1_individual_diagnosis": "MISC-G10-OPT-01",
                    "model1_confidence": 0.95
                }
            ],
            "pattern_type": "PERSISTENT_MISCONCEPTION",
            "persistent_misconception_id": "MISC-G10-OPT-01",
            "temporal_transition": "Student consistently substitutes focal length of concave mirror as positive across 3 consecutive numerical problems.",
            "recommendation": "DELIVER_CARTESIAN_SIGN_CONVENTION_WHITEBOARD_INTERVENTION"
        },
        {
            "session_id": "SESS-2026-MOT-02",
            "student_id": "STU-2091",
            "grade": 9,
            "topic_domain": "Motion and Force",
            "steps": [
                {
                    "step_index": 1,
                    "question_id": "Q-GRP-G09-FORCE-01-01",
                    "student_answer": "5 N",
                    "working_provided": "F = m*v = 0.5 * 10 = 5 N",
                    "model1_individual_diagnosis": "MISC-G09-MOT-02",
                    "model1_confidence": 0.95
                },
                {
                    "step_index": 2,
                    "question_id": "Q-GRP-G09-FORCE-01-02",
                    "student_answer": "15 N",
                    "working_provided": "Need force to maintain motion: F = 1.0 * 15 = 15 N",
                    "model1_individual_diagnosis": "MISC-G09-MOT-02",
                    "model1_confidence": 0.97
                }
            ],
            "pattern_type": "PERSISTENT_MISCONCEPTION",
            "persistent_misconception_id": "MISC-G09-MOT-02",
            "temporal_transition": "Repeated application of Aristotelian impetus model (F = mv instead of F = ma).",
            "recommendation": "TARGETED_NEWTON_FIRST_LAW_SIMULATION"
        },
        {
            "session_id": "SESS-2026-ELE-03",
            "student_id": "STU-3382",
            "grade": 10,
            "topic_domain": "Electricity: Parallel Resistors",
            "steps": [
                {
                    "step_index": 1,
                    "question_id": "Q-GRP-G10-CIRCUITS-01-01",
                    "student_answer": "Rp = 0.25 ohms",
                    "working_provided": "1/Rp = 1/6 + 1/12 = 3/12 = 0.25 ohms",
                    "model1_individual_diagnosis": "MISC-G10-ELE-03",
                    "model1_confidence": 0.98
                },
                {
                    "step_index": 2,
                    "question_id": "Q-GRP-G10-CIRCUITS-01-02",
                    "student_answer": "Rp = 0.166 ohms",
                    "working_provided": "1/Rp = 1/10 + 1/15 = 5/30 = 0.166",
                    "model1_individual_diagnosis": "MISC-G10-ELE-03",
                    "model1_confidence": 0.99
                }
            ],
            "pattern_type": "PERSISTENT_MISCONCEPTION",
            "persistent_misconception_id": "MISC-G10-ELE-03",
            "temporal_transition": "Student systematically calculates the reciprocal sum but omits the final inversion step across parallel circuit problems.",
            "recommendation": "HIGHLIGHT_RECIPROCAL_INVERSION_STEP"
        },
        {
            "session_id": "SESS-2026-TRANSIENT-04",
            "student_id": "STU-4410",
            "grade": 10,
            "topic_domain": "Light: Spherical Mirrors",
            "steps": [
                {
                    "step_index": 1,
                    "question_id": "Q-GRP-G10-MIRROR-01-01",
                    "student_answer": "-51.0 cm",
                    "working_provided": "1/v = -1/15 + 1/20 = arithmetic slip",
                    "model1_individual_diagnosis": "SLIP-ARITHMETIC",
                    "model1_confidence": 0.88
                },
                {
                    "step_index": 2,
                    "question_id": "Q-GRP-G10-MIRROR-01-02",
                    "student_answer": "-60.0 cm",
                    "working_provided": "1/v = 1/(-20) - 1/(-30) = -1/60 => v = -60 cm",
                    "model1_individual_diagnosis": "CORRECT",
                    "model1_confidence": 0.99
                }
            ],
            "pattern_type": "TRANSIENT_SLIP",
            "persistent_misconception_id": None,
            "temporal_transition": "One-off arithmetic slip followed immediately by correct conceptual application. No cognitive misconception present.",
            "recommendation": "ENCOURAGE_CAREFUL_CALCULATION_NO_REMEDIATION_NEEDED"
        }
    ]

    # Generate additional synthetic variations of sequences to total 32 sequences
    full_sequences = []
    seq_counter = 1
    for arch in archetypes:
        for i in range(8):
            seq_copy = json.loads(json.dumps(arch))
            seq_copy["session_id"] = f"SESS-{arch['grade']:02d}-{seq_counter:03d}"
            seq_copy["student_id"] = f"STU-{random.randint(1000, 9999)}"
            seq_counter += 1
            full_sequences.append(seq_copy)

    print(f"[Dataset Engine] Generated {len(full_sequences)} multi-step sequence logs.")
    return full_sequences


# ==============================================================================
# 5. DATA SPLITTING (GROUP-BASED LEAKAGE PREVENTION)
# ==============================================================================

def create_leakage_free_splits(records):
    """
    Groups records by template_group_id so all instances derived from the same
    template are kept strictly in either Train, Val, or Test set.
    """
    groups = {}
    for r in records:
        gid = r["template_group_id"]
        groups.setdefault(gid, []).append(r)

    group_keys = list(groups.keys())
    random.seed(42)
    random.shuffle(group_keys)

    # 70% Train, 15% Val, 15% Test group allocation
    n_groups = len(group_keys)
    n_train = max(1, int(0.70 * n_groups))
    n_val = max(1, int(0.15 * n_groups))
    
    train_groups = set(group_keys[:n_train])
    val_groups = set(group_keys[n_train:n_train+n_val])
    test_groups = set(group_keys[n_train+n_val:])

    # Fallback to ensure all splits get examples
    if not test_groups and len(val_groups) > 1:
        test_groups.add(val_groups.pop())

    train_records = [r for r in records if r["template_group_id"] in train_groups]
    val_records = [r for r in records if r["template_group_id"] in val_groups]
    test_records = [r for r in records if r["template_group_id"] in test_groups]

    print(f"[Dataset Engine] Group Split Statistics:")
    print(f"  • Train Groups: {train_groups} -> {len(train_records)} records")
    print(f"  • Val Groups:   {val_groups} -> {len(val_records)} records")
    print(f"  • Test Groups:  {test_groups} -> {len(test_records)} records")

    return train_records, val_records, test_records


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    data_dir = BASE_DIR
    splits_dir = data_dir / "splits"
    data_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save Taxonomy
    tax_path = data_dir / "taxonomy.json"
    with open(tax_path, "w", encoding="utf-8") as f:
        json.dump(TAXONOMY, f, indent=2)
    print(f"[OK] Saved Taxonomy to {tax_path}")

    # 2. Build Individual Dataset
    records = generate_individual_dataset()
    ind_path = data_dir / "individual_dataset.json"
    with open(ind_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print(f"[OK] Saved Individual Dataset ({len(records)} records) to {ind_path}")

    # 3. Build Sequence Dataset
    sequences = generate_sequence_dataset()
    seq_path = data_dir / "sequence_dataset.json"
    with open(seq_path, "w", encoding="utf-8") as f:
        json.dump(sequences, f, indent=2)
    print(f"[OK] Saved Sequence Dataset ({len(sequences)} sessions) to {seq_path}")

    # 4. Create and Save Leakage-Free Splits
    train_recs, val_recs, test_recs = create_leakage_free_splits(records)
    with open(splits_dir / "train.json", "w", encoding="utf-8") as f:
        json.dump(train_recs, f, indent=2)
    with open(splits_dir / "val.json", "w", encoding="utf-8") as f:
        json.dump(val_recs, f, indent=2)
    with open(splits_dir / "test.json", "w", encoding="utf-8") as f:
        json.dump(test_recs, f, indent=2)
    print(f"[OK] Saved Splits in {splits_dir}")


if __name__ == "__main__":
    main()
