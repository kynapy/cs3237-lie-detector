# CS3237 Lie Detector

## About This Project
This is a Lie Detector created for the CS3237 Introduction to Internet of Things module in the National University of Singapore (NUS). This IoT project uses measurements of biological features such as heart rate and facial features to identify if a person is lying. The data is uploaded to a cloud server, where a machine learning model is used to predict if the person is lying. This project aims to be an alternative to polygraph tests, which are invasive and require an experienced interviewer to conduct. With our lie detector, we aim to replace the need for experienced personnel with a machine learning model.

## Sensors and Actuators Required
The project has been built and tested with the KY-039 Heartbeat Sensor module, KY-012 Active Buzzer module, KY-004 Button Key Switch module and a builtin camera of a laptop. On top of that, 2 WeMos D1 R2 devices are used as IoT devices in this project. 

## Code Organisation
The code used for the project is in the `/complete_project` directory, with the code for the IoT devices in the `/wemos` sub-directory and the camera and main code in the `/python` sub-directory. The code used for data collection for training of the machine learning model is in the `/ml_training` directory.

## External Links
1. [Project Proposal](https://docs.google.com/document/d/1GnE9zQQFZGiXMbUGtiYWFBUu-CAOxkHn/edit?usp=sharing&ouid=109384428354644706956&rtpof=true&sd=true)
2. [Project Report](https://docs.google.com/document/d/19EIODqlSxa6WgW19Ab_0fNh2WY51sCD-WKREVFHCPhA/edit?usp=sharing)
3. [Project Demo]() (Will be updated)

<br>

## References
1. [Gonzalez-Billandon, J., Aroyo, A. M., Tonelli, A., Pasquali, D., Sciutti, A., Gori, M., Sandini, G., Rea, F. (2019). Can a robot catch you lying? A machine learning system to detect lies during interactions. Frontiers in Robotics and AI, 6.](https://doi.org/10.3389/frobt.2019.00064)

2. [Mehendale, N. (2020). Facial emotion recognition using convolutional neural networks (FERC). SN Applied Sciences, 2(3).](https://doi.org/10.1007/s42452-020-2234-1)

3. [Nurçin, F. V., Imanov, E., Işın, A., & Ozsahin, D. U. (2017). Lie detection on pupil size by back Propagation Neural Network. Procedia Computer Science, 120, 417–421.](https://doi.org/10.1016/j.procs.2017.11.258)