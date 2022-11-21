Fourier_PINN
===========================
This document is used as a detailed description of the Fourier_PINN project.

****

Demo: Toggle Model

https://user-images.githubusercontent.com/90367338/203161015-9bf847e1-f799-4e6b-a4ce-567d2aed3fa4.mp4

****

Demo: Repressilator Model


https://user-images.githubusercontent.com/90367338/203161074-18fa2ee5-a2d8-415c-8a52-472e623541a6.mp4


****

Demo: Lorenz Model

https://user-images.githubusercontent.com/90367338/195965938-8b97b3d6-3039-4a14-985e-099ce1090c0b.mp4


****

Demo: Predator–Prey Model

https://user-images.githubusercontent.com/90367338/195965911-765e679c-3ead-4015-a841-d9d955cbf68f.mp4

****

Demo: Compartmental Model in Epidemiology (SIR Model)

https://user-images.githubusercontent.com/90367338/196823671-819993fe-cf2a-4891-97a4-862b692005b3.mp4

****

Demo: Turing Pattern Model

https://user-images.githubusercontent.com/90367338/196028449-c348fb8b-ff31-4a78-96a0-d850bf9e0774.mp4


****
 
| Project Name | Fourier_PINN |
|--------------|--------------|
| Author       | Enze Xu      |
| Version      | v1.3.0       |

****
# Catalogue
* [Introduction to files](#introduction-to-files)
* [Start](#start)
* [Contact](#contact)

****
# Introduction to files
1. config_{XX}.py: XX model configurations
2. model_{XX}.py: XX model class
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
(venv) $ python3 model_Turing.py
```

Exit virtual environment
```shell
(venv) $ deactivate
```
****

# Contact
If you have any questions, suggestions, or improvements, please get in touch with xue20@wfu.edu
****
