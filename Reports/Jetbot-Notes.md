# Jetbot Notes - Francisco

## Jetbot Setup and Configuration

- **Wi-Fi Configuration and Date/Time Setup**:
    - Powered on the Jetson Nano and reset the date and time to ensure proper Wi-Fi functionality.
    - Installed Python 3.6.9 on the Jetson Nano.
    - Ensured the Jetson Nano was connected to the Pomona Network for Wi-Fi access:
        - For Wi-Fi connection, the Jetbot must connect to the second available Pomona Network option.
    - Resolved date/time mismatch causing repository errors:
        - Corrected the device date, which was off by a month, to fix repository errors.
        - Updated Ubuntu on the Jetbot, upgraded the firmware, and installed necessary Wi-Fi drivers to get Wi-Fi working on the Jetbot:
            ```bash
            sudo nmtui
            ```
    - Successfully established an SSH connection for remote access:
        ```bash
        ssh -L 8001:localhost:8001 <username>@<hostname>
        ```

## Hardware Troubleshooting

- **Power and Ethernet Connection**:
    - Connected the barrel jack to the Jetson Nano B01 board. Ensured the developer kit’s total load exceeded 2A by connecting the J48 Power Select Header pins to disable power supply via Micro-USB and enable 5V⎓4A via the J25 power jack. Another option was to supply 5V⎓5A via the J41 expansion header (2.5A per pin).
    - Initially, the board appeared to have 3 pins, but there were only 2 with the jumper hanging off one pin. Moved the jumper down to cover both pins and got the barrel jack power working.
    - Recommended shutting down the computer and unplugging the power when connecting peripherals.
    - Connected the ethernet cable; faced initial issues with internet connection. Changed the ethernet port connected to the Jetson Nano and successfully activated the ethernet connection. Faced issues with Wi-Fi repeatedly disconnecting, requiring registration on Pomona Network.

## Docker Setup and Issues

- **Docker Setup Attempts**:
    - Attempted to configure Docker for online programming across devices, but encountered persistent "image not found" errors.
    - Shifted focus to running Python files directly from the terminal, but encountered issues with the camera not being read correctly by OpenCV. Verified the camera hardware was functioning using the command `nvgstcapture-1.0`.
    - Followed [this Docker setup tutorial](https://jetbot.org/master/software_setup/docker.html) for the Jetson Nano:
        - Username: ARCS

## OpenCV and GStreamer Issue

- **OpenCV and GStreamer**:
    - Researched and addressed the 'Illegal instruction (core dumped)' error encountered when using OpenCV with GStreamer.
    - References:
        - [NVIDIA Developer Forum](https://forums.developer.nvidia.com/t/opencv-video-capture-with-gstreamer-doesnt-work-on-ros-melodic/113147)
        - [Stack Overflow link](https://stackoverflow.com/questions/65631801/illegal-instructioncore-dumped-error-on-jetson-nano)

## Qwiic Driver Issues

- **Qwiic Motor Driver**:
    - Encountered significant problems with importing `qwiic`, spending 3 hours troubleshooting:
        - Error persisted despite multiple restarts, reinstallations, and attempts to install locally from GitHub.
        - Determined that the Jetbot was not recognizing some of its hardware components. Specifically, the Sparkfun Qwiic motor driver was not recognized. After downloading additional libraries, the motor driver controller was recognized, but the motors connected to it were not.
    - Created a new environment named `jetbot_env`.

## Documentation and Setup

- **Documenting Jetbot Setup**:
    - To log in to the Jetbot, use the credentials found in the appropriate document.
    - Ensure the date and time are set correctly. Useful resources for the Jetbot:
        - [First Picture CSI-USB Camera](https://developer.nvidia.com/embedded/learn/tutorials/first-picture-csi-usb-camera)
        - [Jetson Inference Streaming](https://github.com/dusty-nv/jetson-inference/blob/master/docs/aux-streaming.md#v4l2-cameras)

## Future Plans

- **Future Plans**:
    - Improve the accuracy and consistency of blinking detection.
    - Develop a Python script for data communication between Python scripts.
    - Deploy eye tracking data script on Jetson Nano and set up a live feed from the robot.
    - Get motors working through Python commands on Jetson Nano.

## Additional Notes

- **Jetson Nano Setup**:
    - Don't plug in the Jetbot directly into USB for power.
    - Ensure correct connection of the J48 Power Select Header pins to use the barrel jack power supply.
    - In case of Wi-Fi disconnection, ensure to register the device on Pomona Network via Network Access Control.

- **Commands**:
    - To check if the camera is working:
        ```bash
        nvgstcapture-1.0
        ```
    - To configure network connections:
        ```bash
        sudo nmtui
        ```
    - To establish SSH connection:
        ```bash
        ssh -L 8001:localhost:8001 <username>@<hostname>
        ```