# simple-retarget

# Acknowledgement
This repo is insipired by and adopted from the [PHC](https://github.com/ZhengyiLuo/PHC) codebase. The implementation provided here is refactored to be cleaner and more readable for learning purpose.

1. Clone this repo and install dependencies
   ```
    git clone git@github.com:ZhengyiLuo/PHC.git # ssh recommended
    pip install -r requirements.txt
   ```
2. Download SMPL paramters from [SMPL](https://smpl.is.tue.mpg.de/) and put them under `data/smpl`:

    ```
    |-- data
        |-- smpl
            |-- SMPL_FEMALE.pkl
            |-- SMPL_NEUTRAL.pkl
            |-- SMPL_MALE.pkl

    ```