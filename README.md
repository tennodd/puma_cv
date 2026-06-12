# puma_cv

Підсистема технічного зору для робота-маніпулятора PUMA 560 (командний проєкт).
Виявляє об'єкти на робочій поверхні за допомогою сенсора Kinect v1 та публікує
результати в ROS 2.

Реалізовано на ROS 2 Jazzy та Python 3.12. Обробка зображень — OpenCV 4.6 і
Open3D 0.19, робота з Kinect v1 - через libfreenect.

Репозиторій містить три пакети: `cv_interfaces` з визначенням повідомлення
`Detection.msg`, `cv_webcam_demo` з власне вузлом детектора та `kinect_ros2` —
драйвер Kinect v1 для ROS 2.

## Збірка та запуск

Усі три пакети потрібно розмістити в каталозі `src/` робочого простору ROS 2,
після чого:

```bash
# Встановлення Open3D
bash setup_container_deps.sh

# Збірка
colcon build --packages-select cv_interfaces cv_webcam_demo kinect_ros2 --symlink-install
source install/setup.bash

# Запуск
ros2 launch cv_webcam_demo webcam_cv.launch.py
```

Результати детекції публікуються в топік `/detections` (тип
`cv_interfaces/msg/Detection`).

## Налаштування

Усі параметри детектора доступні для зміни під час роботи через стандартний
механізм параметрів ROS 2 - зручніше за все через `rqt_reconfigure` або
`ros2 param set`. Для перегляду відеопотоків у реальному часі можна
скористатися `rqt`.
