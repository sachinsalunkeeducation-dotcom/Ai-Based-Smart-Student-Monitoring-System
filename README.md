# Ai-Based-Smart-Student-Monitoring-System

## Overview

The AI-Based Smart Student Monitoring System is an intelligent surveillance and discipline management solution designed for educational institutions. The system automatically monitors student discipline, tracks attendance, detects unauthorized individuals, and provides real-time insights through a web dashboard.

## Key Features

-Student Discipline Monitoring
-Detects whether a student's shirt is properly tucked in.
-Verifies whether the student is wearing the correct college uniform.
-Detects the presence of a valid college ID card.
-Classifies students as disciplined or non-disciplined.

## Attendance Management

-Automatically records student attendance within the campus.
-Uses face recognition for student identification.
-Maintains attendance records in the database.

## Unknown Person Detection

-Detects unauthorized or unknown individuals entering the campus.
-Captures evidence images of unknown persons.
-Sends detected evidence to the authorized personnel through the Unknown Section.

## Dashboard and Analytics

-Displays student information on the frontend dashboard.
-Shows attendance status.
-Displays discipline status.
-Provides monitoring information to authorized users and administrators.

## Technologies Used

Artificial Intelligence & Computer Vision
YOLOv8n Pose Model
Custom YOLO Model for ID Card Detection
InsightFace
OpenCV

# Backend
-Python
-SQLite Database
# Frontend
-React.js
-Tailwind CSS

## System Workflow

1.Camera captures student video streams.
2.YOLOv8 detects students and analyzes discipline-related attributes.
3.InsightFace identifies students using face recognition.
4.Custom-trained YOLO model detects college ID cards.
5.Attendance and discipline records are stored in SQLite.
6.Unknown individuals are detected and logged with evidence.
7.Results are displayed on the frontend dashboard in real time.

## Future Enhancements

-Email/SMS alerts for student parents
-Detailed analytics and reporting
-Multi-camera support
-Cloud database integration
