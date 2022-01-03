# Using Anaconda

## Installing Anaconda

**Last revised:** 24-Dec-2021 by Dick Furnstahl (furnstahl.1@osu.edu)

**Overview of what you need: Anaconda Python**

You may already have Python on your computer, but it may not be the version we need (Python changes rapidly) and there are many additional software libraries we will need for scientific computing: Numpy, Scipy, Matplotlib are examples. Fortunately, there is a one-stop-shopping solution: download and install Anaconda Python. It comes with all of the packages we need, plus a great mechanism (a package manager called conda) for updating or installing new packages. Anaconda is available for Windows, macOS, and Linux. One caveat is that the package is not small: the size is almost 3GB. If that is a problem, ask about other options to installing the full package (see the documentation on Miniconda).

**How to get Anaconda Python** **[Also see the Anaconda Distribution Starter Guide]**

Step-by-step instructions for downloading and installing Anaconda are available for [Windows](http://docs.anaconda.com/anaconda/install/windows/), [Mac OSX](http://docs.anaconda.com/anaconda/install/mac-os/), or [Linux](http://docs.anaconda.com/anaconda/install/linux/). Here we&#39;ll step through the relevant options for us (based on following the graphical installer, which is what you will get by default for Windows and Macs; for Linux it is assumed you can follow the [instructions for Linux](http://docs.anaconda.com/anaconda/install/linux/)).

1. Go to [https://www.anaconda.com/products/individual - Downloads](https://www.anaconda.com/products/individual#Downloads) and click on the **Download** button for Python 3.9 for your type of computer (use the 64-bit installer).
2. An installation program will be downloaded. Start it up (e.g., by double clicking).
3. Accept the license agreement.
4. Install for All Users (this assumes you have admin privileges; it avoids later problems if you can do this). Allow the installation if your virus protection software asks you.
5. It is suggested you accept the suggestion for the installation location (e.g., C:\ProgramData\Anaconda3 on Windows). [Note: steps 4 and 5 may be reversed for a Mac install.]
6. For Windows, accept the checked Advanced Options as &quot;Register Anaconda as my default Python 3.8&quot; and click **Install**.
7. It is recommended (for our purposes) that you &quot;Skip&quot; installing Microsoft VSCode, if you are given the option.
8. While it is installing, you might want to click **Show Details** to see progress. It will take a while, depending on your hardware. Be patient!
9. If it works, you&#39;ll get to a &quot;Thanks for installing Anaconda Individual Edition!&quot; page. There will be two already-checked boxes. You _ **do not** _ need to sign up for Anaconda Cloud, so uncheck that one. But you might want to read the &quot;Getting started with Anaconda&quot; information. Click **Finish** you are done.

## Anaconda and github

**Installation of the Physics 5300 notes and Jupyter notebooks from GitHub by command line**

Go to the location on your computer where you want the notebook files. 

Download the 5300 repository from GitHub [https://github.com/furnstahl/5300-JB](https://github.com/furnstahl/5300-JB) and enter the directory `5300-JB` which should have been created on your computer with the following:

```
git clone https://github.com/furnstahl/5300-JB.git
cd 5300-JB
```

The notebooks that we will be using depend on several scientific python modules (see the list in environment.yml) and require a python3.9 installation based on Anaconda. This is best done within a conda environment, as described in the next section.


## Anaconda environments

### Creating a conda environment

The python modules needed for 5300 and their dependencies are best installed using ``conda`` by creating
a virtual environment:

	conda env create

which reads the `environment.yml` file in your current directory (in this case the one in 5300-JB). [Note: if you are using Windows, you should do this in an Anaconda Prompt window.]

To use a different file name such as `other_environment.yml`: `conda env create -f other_environment.yml`

It will take a while to generate the new environment; be patient!  You will see a listing being generated like:
        
       Downloading and Extracting Packages
       scikit-learn-0.21.3  | 5.9 MB    | ##################################### | 100% 
       glib-2.58.3          | 3.1 MB    | ##################################### | 100% 
       libcxx-8.0.1         | 1000 KB   | ##################################### | 100% 
       scipy-1.3.1          | 18.1 MB   | ##################################### | 100% 
etc. (it will be a pretty long list with different version numbers and not necessarily in this order).  Then you'll see

       Preparing transaction: done
       Verifying transaction: done
       Executing transaction: done 
where "done" appears when it has finished.  If all is well at the end, you'll get a success message.  If it fails, email the message you get to furnstahl.1@osu.edu and we'll figure out the problem.

Some packages might not be found in the default conda channels. One
can either specify relevant package channel(s) in the environment.yml
file (as done here), or add them to the default conda channel configuration via, e.g,

	conda config --append channels conda-forge

You shouldn't need to do this for the initial setup.

Once the virtual environment has been created it can be activated (the name "5300-env" was specified in the environment.yml file):

    conda activate 5300-env

To deactivate the virtual environment:

    conda deactivate

Note that earlier versions of conda used 'source' instead of 'conda'
to activate environments. If you have an earlier version of conda you
might have to do

    conda update conda

Note that there are also other options ('venv', 'pipenv') for creating virtual
environments that includes the python version and packages that we will be using.

Once the environment is set up and activated, you are encouraged to run the test Jupyter notebook:

    cd LectureNotes/notebooks
    jupyter notebook Anaconda_Bayes_test.ipynb

Other notebooks can be found in the subdirectories of the notebooks directory.

### Updating your conda environment for 5300

Go to the `5300-JB` directory you created by cloning the class repository from GitHub.  This is where the relevant `environment.yml` file is stored.  This file defines the environment and will be occasionally updated to add additional modules.  
You can update to the new environment with:

    conda deactivate
    conda env update 
    
Now if you activate the environment again:

    conda activate 5300-env

you will have access to the new packages.

### Changing to the `5300-env` env kernel when running a Jupyter notebook

If you are running the `5300-env` kernel, you should see

    Python [conda env:5300-env] *

in the upper right-hand corner of the Jupyter notebook you are running.  If it just says something like `Python 3` then you are not running the `5300-env` kernel.  In that case, look under the `Kernel` menu to `Change kernel` and select `Python [conda env:5300-env]`.  The kernel should restart and indicate the new kernel in the upper corner.
