Fourier_PINN
===========================
This document is used to show a detailed description of the Fourier_PINN project.

****
Demo: PP Model

https://user-images.githubusercontent.com/90367338/195965911-765e679c-3ead-4015-a841-d9d955cbf68f.mp4

****

Demo: Lorenz Model

https://user-images.githubusercontent.com/90367338/195965938-8b97b3d6-3039-4a14-985e-099ce1090c0b.mp4


****
 
| Project Name | Fourier_PINN |
|--------------|--------------|
| Author       | Enze Xu      |
| Version      | v1.2.0       |

****
# Catalogue
* [Introduction to files](#introduction-to-files)
* [Start](#start)
* [Contact](#contact)

****
# Introduction to files
1. config.py: model configurations
2. model.py: model class
3. utils.py: functions for plotting

****
# Start
See `https://github.com/EnzeXu/Fourier_PINN` or
```shell
$ git clone https://github.com/EnzeXu/Fourier_PINN.git
```

Set virtual environment and install packages: (Python 3.7+ (3.7, 3.8, 3.9 or higher) is preferred)
```shell
$ python3 -m venv ./venv
$ source venv/bin/activate
(venv) $ pip install -r requirements.txt
```

Run
```shell
(venv) $ python3 model_PP.py
(venv) $ python3 model_Lorenz.py
```

Exit virtual environment
```shell
(venv) $ deactivate
```
****

# Contact
If you have any questions, suggestions, or improvements, please get in touch with xue20@wfu.edu
****
