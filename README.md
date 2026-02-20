# BHARAT-AI-SOC-CHALLENGE-PROBLEM-STATEMENT-2-
An edge-AI powered gesture control system that uses real-time hand tracking and multi-user face authentication to securely control media playback without physical interaction. Built for NVIDIA Jetson platforms, it performs low-latency on-device inference with lock/unlock security and system-level VLC integration.

Run capture.py first to capture and store authorized user images in the pro/ directory for face authentication.
Then execute hci.py to start the secure gesture control system, which grants access only to users whose images are registered in pro/.
