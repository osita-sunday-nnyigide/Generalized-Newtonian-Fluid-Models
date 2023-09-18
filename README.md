This program is used for fitting experimental data to GNF models.
Currently, the following models are implemented: 

`Ellis`

`Sisko`

`Williamson`

`Cross`

`Power Law`

`Carreau-Yasuda`

#Installation of dependencies

`pip install tk`

`pip install scipy`

`pip install numpy`

`pip install matplotlib`

## Usage
Dowload this ZIP and copy the folder, Generalized-Newtonian-Fluid-Models, to any location where you intend to run this program. On windows, double-click FitPro.bat to open the GUI.
On a linux machine, cd to the Generalized-Newtonian-Fluid-Models folder and open a linux bash. On command prompt, enter python3  ModelFitting.py to run the GUI.
As an example, a text file named data.txt or data.dat has been provided. Upload this data, select Carreau-Yasuda Model and click submit to fit using default fitting parameters.
For the second example, upload dna.dat and select Power Law to fit using default parameters. Like any other fitting software, if the default parameters are not suitable for the selected model, the program will generate a warning message.


## License
[MIT](https://choosealicense.com/licenses/mit/)

