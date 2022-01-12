#!/usr/bin/env python
# coding: utf-8

# # Python and Jupyter notebooks: part 01
# 
# Last revised: 31-Dec-2021 by Dick Furnstahl [furnstahl.1@osu.edu]

# **You can find valuable documentation under the Jupyter notebook Help menu. The "User Interface Tour" and "Keyboard Shortcuts" are useful places to start, but there are also many other links to documentation there.** 
# 
# *Select "User Interface Tour" and use the arrow keys to step through the tour.*

# This is a whirlwind tour of just the minimum we need to know about Python and Jupyter notebooks to get started doing mechanics problems.  We'll explore additional features and details as we proceed.
# 
# A Jupyter notebook is displayed on a web browser running on a computer, tablet (e.g., IPad), or even your smartphone.  The notebook is divided into *cells*, of which two types are relevant for us:
# * **Markdown cells:** These have headings, text, and mathematical formulas in $\LaTeX$ using a simple form of HTML called *markdown*.
# * **Code cells:** These have Python code (or other languages, but we'll stick to Python).
# 
# Either type of cell can be selected with your cursor and will be highlighted in color on the left when active.  You evaluate an active cell with shift-return (as with Mathematica) or by pressing `Run` on the notebook toolbar.  Some notes:
# * When a new cell is inserted, by default it is a Code cell and will have `In []:` to the left.  You can type Python expressions or entire programs in a cell.  How you break up code between cells is your choice and you can always put Markdown cells in between.  When you evaluate a cell it advances to the next number, e.g., `In [5]:`.
# * On the notebook menu bar is a pulldown menu that lets you change back and forth between Code and Markdown cells.  Once you evaluate a Markdown cell, it gets formatted (and has a blue border).  To edit the Markdown cell, double click in it. 
# 
# **Try double-clicking on this cell and then shift-return.**  You will see that a bullet list is created just with an asterisk and a space at the beginning of lines (without the space you get *italics* and with two asterisks you get **bold**).  **Double click on the title header above and you'll see it starts with a single #.**  Headings of subsections are made using ## or ###.  See this [Markdown cheatsheet](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet) for a quick tour of the Markdown language (including how to add links!).
# 
# **Now try turning the next (empty) cell to a Markdown cell and type:** `Einstein says $E=mc^2$` **and then evaluate it.**  This is $\LaTeX$! (If you forget to convert to Markdown and get `SyntaxError: invalid syntax`, just select the cell and convert to Markdown with the menu.)

# In[ ]:





# The notebook menus enable you to rename your notebook file (always ending in `.ipynb`) or `Save and Checkpoint` to save the changes to your notebook.  You can insert and delete cells (use the up and down arrows in the toolbar to easily move cells).  You will often use the `Kernel` menu to `Restart` the notebook (and possibly clear output).

# As you get more proficient working with notebooks, you will probably start using the shortcut keys from the command mode of cells. A cell that is marked in blue implies that you are in command mode. You can start editing the cell by hitting `Enter` (or by clicking inside it). You can exit from edit mode into command mode by hitting `Esc`. A list of shortcut keys can be seen when opening the command palette by clicking on the keyboard button.

# ## Ok, time to try out Python expressions and numpy
# 
# We can use the Jupyter notebook as a super calculator much like Mathematica and Matlab.  **Try some basic operations, modifying and evaluating the following cells, noting that exponentiation is with** `**` **and not** `^`.

# In[ ]:


1 + 1  # Everything after a number sign / pound sign / hashtag) 
       #  is a comment


# In[ ]:


# Try other examples here. The spaces are optional but help with readability.
3.2 * 4.713


# Note that if we want a floating point number, which will be the same precision as a `double` in C++, we *usually* include a decimal point (even when we don't have to) while a number without a decimal point is an integer. (Note: there is no problem with 1/2 being evaluated as zero, as in C++.)

# In[ ]:


3.**2


# We can define integer, floating point, and string variables, perform operations on them, and print them.  Note that we don't have to predefine the type of a variable and we can use underscores in the names (unlike Mathematica).  **Evaluate the following cells and then try your own versions.** 

# In[ ]:


x = 5.
print(x)
x   # If the last line of a cell returns a value, it is printed with Out[#].


# In[ ]:


y = 3.*x**2 - 2.*x + 7.
print('y = ', y)           # Strings delimited by ' 's


# There are several ways to print strings that includes variables from your code. We recommend using `fstring`. See, e.g., this [blog](https://miguendes.me/73-examples-to-help-you-master-pythons-f-strings) for examples. Can you deduce what `.0f` and `.2f` mean in the following examples? 

# In[ ]:


print(f'y = {y:.0f}')      # Just a preview: more on format later 
print(f'y = {y:.2f}')      #  (note that this uses the "new" fstring)


# We will mostly use `fstring` in this course, but you might also encounter older formatting syntax:

# In[ ]:


print('x = %.2f  y = %.2f' %(x,y)) 
print('x = {0:.2f}  y = {1:.2f}'.format(x, y)) 
# compare to the fstring version
print(f'x = {x:.2f}  y = {y:.2f}')


# **Change the following to print your own name.**

# In[ ]:


first_name = 'Dick'     # Strings delimited by ' 's
last_name = 'Furnstahl'
full_name = first_name + ' ' + last_name  # you can concatenate strings 
print(full_name)
# or
print(f'{first_name} {last_name}')


# Ok, what about square roots and trigonometric functions and ... 
# 
# *(Note: the next cells will give error messages --- keep reading to see how to fix them.)*

# In[ ]:


sqrt(2)


# In[ ]:


sin(pi)


# We need to `import` these functions through the numpy library. There are other choices, but numpy works with the arrays we will use.  **Note:** *Never* use `from numpy import *` instead of `import numpy as np`.  Here `np` is just a abbreviation for numpy (which we can choose to be anything, but `np` is conventional).

# In[ ]:


import numpy as np


# In[ ]:


print(np.cos(0.))


# Now functions and constants like `np.sqrt` and `np.pi` will work.  **Go back and fix the square root and sine.**

# ### Debugging aside . . .
# 
# Suppose you try to import and it fails (**go ahead and evaluate the cell**):

# In[ ]:


import numpie


# When you get a `ModuleNotFoundError`, the first thing to check is whether you have misspelled the name. **Try using Google, e.g., search for "python numpie".** In this case (and in most others), Google will suggest the correct name (here it is numpy).  If the name does exist, check whether it sounds like the package you wanted.
# 
# If you have the correct spelling, check whether you have installed the relevant package.  If you installed Python with Anaconda (which we will assume you did -- if not, do it!), then use `conda list`, e.g., `conda list numpy` in a Terminal window (on a Mac or Linux box) or in an Anaconda Prompt window (on a Windows PC).

# ### numpy arrays
# 
# We will often use numpy arrays so we'll start with those.  They are *like* lists delimited by square brackets, i.e., `[]`s. To construct an array from `min` to `max` in steps of `step` we can use `np.arange(min, max, step)`.  Examples (**try your own**):

# In[ ]:


t_pts = np.arange(0., 10., .1)
t_pts


# If we give a numpy array to a function, each term in the list is evaluated with that function:

# In[ ]:


x = np.arange(1., 5., 1.)
print(x)
print(x**2)
print(np.sqrt(x))


# We can pick out elements of the list (note the square brackets).  **Why does the last one fail?** 

# In[ ]:


print(x[0])
print(x[3])
print(x[4])


# An alternative to `np.arange` is `np.linspace(min, max, number)` to get an array from `min` to `max` with `number` elements.  Example: 

# In[ ]:


u_pts = np.linspace(0., 10., 10)
u_pts


# **Change the last statement for `u_pts` to get an array from 0 to 10 spaced by 1.**

# ## Getting help
# 
# You will often need help identifying the appropriate Python (or numpy or scipy or ...) command or you will need an example of how to do something or you may get an error message you can't figure out.  In all of these cases, Google (or equivalent) is your friend. Always include "python" in the search string (or "numpy" or "matplotlib" or ...) to avoid getting results for a different language. You will usually get an online manual as one of the first responses if you ask about a function; these usually have examples if you scroll down. Otherwise, answers from *Stack Overflow* queries are your best bet to find a useful answer.

# ## Functions
# 
# There are many Python language features that we will use eventually, but in the short term what we need first are *functions*.  Here we first see the role of *indentation* in Python in place of {}s or ()s in other languages.  We'll always indent four spaces (never tabs!).  We know a function definition is complete when the indentation stops. 
# 
# To find out about a Python function or one you define, put your cursor on the function name and hit `shift+Tab+Tab`. **Go back and try it on `np.arange`.**  

# In[ ]:


# Use "def" to create new functions.  
#  Note the colon in the def line and indentation after that line (4 spaces).
def my_function(x):
    """This function squares the input.  Always include a brief description
        at the top between three starting and three ending quotes.  We will
        talk more about proper documentation later ("docstrings").
        Try shift+Tab+Tab after you have evaluated this function.
    """
    return x**2

print(my_function(5.))

# We can pass an array to the function and it is evaluated term-by-term.
x_pts = np.arange(1.,10.,1.)
print(my_function(x_pts))


# In[ ]:


# Two variables, with a *default* for the second variable
def add(x, y=4.):
    """Add two numbers."""
    print(f"x is {x} and y is {y}")
    return x + y  # Return values with a return statement

# Calling functions with parameters
print('The sum is ', add(5, 6))  # prints "x is 5 and y is 6" and returns 11

# Another way to call functions is with keyword arguments
add(y=6, x=5)  # Keyword arguments can arrive in any order.


# **How do you explain the following result?**

# In[ ]:


add(2)


# ### Debugging aside . . .
# 
# There are *two* bugs in the following function.  **Note the line where an error is first reported and fix the bugs sequentially (so you see the different error messages). You can turn on line numbers with "Toggle Line Numbers" under the View menu.**

# In[ ]:


def hello_function()
    msg = "hello, world!"
    print(msg)
     return msg


# ## Plotting with Matplotlib
# 
# Matplotlib is the plotting library we'll use, at least at first.  We'll follow convention and abbreviate the module we need as `plt`.  The `%matplotlib inline` line tells the Jupyter notebook to make inline plots (note: this is now loaded by default, so we don't need to include that line; we'll see other possibilities later).

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import matplotlib.pyplot as plt


# Procedure we'll use to make a basic plot:
# 
# 0. Generate some data to plot in the form of arrays.
# 1. Create a figure;
# 2. add one or more subplots;
# 3. make a plot and display it.

# In[ ]:


t_pts = np.arange(0., 10., .1)     # step 0.
x_pts = np.sin(t_pts)              # More often this would be from a function 
                                   #  *we* write ourselves.

my_fig = plt.figure()              # step 1.
my_ax = my_fig.add_subplot(1,1,1)  # step 2: rows=1, cols=1, 1st subplot
my_ax.plot(t_pts, x_pts)           # step 3: plot x vs. t


# NOTE: When making just a single plot, you will more usually see steps 1 to 3 compressed into `plt.plot(t_pts, np.sin(t_pts))`.  Don't do this.  It saves a couple of lines but restricts your ability to easily extend the plot, which is what we want to make easy.

# We can always go back and dress up the plot:

# In[ ]:


my_fig = plt.figure()
my_ax = my_fig.add_subplot(1,1,1)  # nrows=1, ncols=1, first plot
# Add color, a line style, and a label to the plot.
my_ax.plot(t_pts, x_pts, color='blue', linestyle='--', label='sine')

my_ax.set_xlabel('t')
my_ax.set_ylabel(r'$\sin(t)$')  # here $s to get LaTeX and r to render it
my_ax.set_title('Sine wave')

# here we'll put the function in the call to plot!
my_ax.plot(t_pts, np.cos(t_pts), label='cosine')  # just label the plot

my_ax.legend();  # turn on legend (labels don't show up until you do this)
                 # try removing the semicolon to see what changes


# Now make two subplots:

# In[ ]:


y_pts = np.exp(t_pts)         # another function for a separate plot

fig = plt.figure(figsize=(10,5))  # allow more room for two subplots

# call the first axis ax1
ax1 = fig.add_subplot(1,2,1)  # one row, two columns, first plot
ax1.plot(t_pts, x_pts, color='blue', linestyle='--', label='sine')
ax1.plot(t_pts, np.cos(t_pts), label='cosine')  # just label the plot
ax1.legend()

ax2 = fig.add_subplot(1,2,2)  # one row, two columns, second plot
ax2.plot(t_pts, np.exp(t_pts), label='exponential')  
ax2.legend()

fig.tight_layout()   # We usually do this to improve the layout of plots


# **Use Google (or whatever) to find how to add x and y labels and try it.**

# ### Saving a figure
# Saving a figure to disk is as simple as calling [`savefig`](http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.savefig) with the name of the file (or a file object). The available image formats depend on the graphics backend you use, but we can always use `png` (which is preferred).

# Let us save the figure (named 'fig') from the previous cell. **Look at the files produced in your browser; if you are using Binder, use `Open...` under the Jupyter notebook `File` menu.**

# In[ ]:


fig.savefig("sine_and_exp.png")
# and a transparent version:
fig.savefig("sine_and_exp_transparent.png", transparent=True)


# ## Solving ordinary differential equations (ODEs)
# 
# Newton's 2nd Law for one particle:
# 
# $$\begin{align}
#   \mathbf{F} = m\mathbf{a} = m\frac{d\mathbf{v}}{dt} = m\frac{d^2\mathbf{x}}{dt^2}
# \end{align}$$ 
# 
# is a first-order differential equation in the velocity vector and a second-order differential equation in the position vector. Here we assume that $\mathbf{F} = \mathbf{F}(\mathbf{x}, \mathbf{v}, t)$ does not have higher derivatives. So we need to know how to solve such differential equations, usually with initial conditions (as opposed to boundary conditions). 
# 
# Here is how to do it with the Scipy function `odeint`.  (Note that `solve_ivp` is now preferred to `odeint`; we'll switch to that later but it is easier to get started with `odeint`.)

# ### First-order ODE

# In[ ]:


# Import the required modules
import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import odeint  # Get only odeint from scipy.integrate
# for the future:
from scipy.integrate import solve_ivp   # Generally preferred to odeint


# Let's try a one-dimensional first-order ODE, say:
# 
# $$\begin{align}
# \quad 
# \frac{dv}{dt} = -g, \quad \mbox{with} \quad v(0) = 10
# \end{align}$$
# 
# in some appropriate units (we'll use MKS units by default).  This ODE can be separated and directly integrated:
# 
# $$\begin{align}
#   \int_{v_0=10}^{v} dv' = - g \int_{0}^{t} dt'
#   \quad\Longrightarrow\quad
#     v - v_0 = - g (t - 0)
#   \quad\Longrightarrow\quad
#    v(t) = 10 - gt
# \end{align}$$
# 
# 

# In[ ]:


# Define a function which calculates the derivative
def dv_dt(v, t, g=9.8):
    """Returns the right side of a simple first-order ODE with default g."""
    return -g   

t_pts = np.linspace(0., 10., 101)     # 101 points from t=0 and t=10.
v_0 = 10.0  # the initial condition
v_pts = odeint(dv_dt, v_0, t_pts)  # odeint( function for rhs, 
                                   #         initial value of v(t),
                                   #         array of t values )


# In[ ]:





# Let's check the output $v(t)$.  **Does it make sense?**

# In[ ]:


print(v_pts[0:10])
v_pts.shape  # This outputs the "shape" of the array: It is two-dimensional!


# Now plot $v(t)$ twice: the numerical calculation and the analytic solution:

# In[ ]:


g = 9.8
v_pts_exact = v_0 - g*t_pts

fig = plt.figure(figsize=(8,4))

ax = fig.add_subplot(1,2,1)
ax.plot(t_pts, v_pts, label='numerical', color='blue', lw=4)
ax.plot(t_pts, v_pts_exact, label='exact', color='red')
ax.set_xlabel('t [sec]')
ax.set_ylabel('v(t) [meters/sec]')
ax.legend();

diff = v_pts_exact - v_pts.flatten()
ax2 = fig.add_subplot(1,2,2)
ax2.semilogy(t_pts, diff, label='difference', color='red')
ax2.set_xlabel('t [sec]')
ax2.set_ylabel('v(t) [meters/sec]')
ax2.legend();

fig.tight_layout()


# We adjusted the colors and linewidths so you could see that the lines are on top of each other. **Why are the differences around $10^{-15}$?**

# **Try solving instead:**
# 
# $$\begin{align}
# \quad 
# \frac{dx}{dt} = -x \quad \mbox{with} \quad x(0) = 1
# \end{align}$$
# 
# **both analytically and numerically.**

# ### Second-order ODE

# Suppose we have a second-order ODE such as:
# 
# $$
# \quad y'' + 2 y' + 2 y = \cos(2x), \quad \quad y(0) = 0, \; y'(0) = 0
# $$
# 
# We can turn this into two first-order equations by defining a new dependent variable equal to $y'$ (we'll call it $z$):
# 
# $$
# \quad z \equiv y' \quad \Rightarrow \quad z' + 2 z + 2y = \cos(2x) 
#   \quad\mbox{or}\quad z' = -2z -2y + \cos(2x), \quad\mbox{with}
# \quad z(0)=y(0) = 0.
# $$
# 
# Now introduce the vector 
# 
# $$
#   \mathbf{U}(x) = \left(\begin{array}{c}
#                          y(x) \\
#                          z(x)
#                         \end{array}
#                   \right)
#         \quad\Longrightarrow\quad
#     \frac{d\mathbf{U}}{dx} = 
#                              \left(\begin{array}{c}
#                                     y' \\
#                                     z'
#                                    \end{array}
#                              \right) 
#                            =  
#                              \left(\begin{array}{c}
#                                     z \\
#                                     -2 y' - 2 y + \cos(2x)
#                                    \end{array}
#                              \right)
#               \quad\mbox{with}\quad
#     \mathbf{U}(0) = \left(\begin{array}{c}
#                          0 \\
#                          0
#                         \end{array}
#                   \right)
# $$
# 
# We can solve this system of ODEs using `odeint` with lists, as follows:

# In[ ]:


# Define a function for the right side
def dU_dx(U, x):
    """Right side of the differential equation to be solved.
    U is a two-component vector with y=U[0] and z=U[1]. 
    Thus this function should return [y', z']
    """
    return [U[1], -2*U[1] - 2*U[0] + np.cos(2*x)]

# initial condition U_0 = [y(0)=0, z(0)=y'(0)=0]
U_0 = [0., 0.]

x_pts = np.linspace(0, 15, 200)  # Set up the mesh of x points
U_pts = odeint(dU_dx, U_0, x_pts)  # U_pts is a 2-dimensional array
y_pts = U_pts[:, 0]   # Ok, this is tricky.  For each x, U_pts has two 
                      #  components.  We want the upper component for all
                      #  x, which is y(x).  The : means all of the first 
                      #  index, which is x, and the 0 means the first
                      #  component in the other dimension.
                      # What is U_pts[:, 1]? (Try plotting it.)


# In[ ]:


fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(x_pts, y_pts)
ax.set_xlabel('x')
ax.set_ylabel('y')


# ## Widgets!
# 
# A widget is an object such as a slider or a check box or a pulldown menu.  We can use them to make it easy to explore different parameter values in a problem we're solving, which is invaluable for building intuition.  They act on the argument of a function.  We'll look at a simple case here but plan to explore this more as we proceed.
# 
# The set of widgets we'll use here (there are others!) is from `ipywidgets`; we'll conventionally import the module as `import ipywidgets as widgets` and we'll also often use `display` from `Ipython.display`.

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

import ipywidgets as widgets
from IPython.display import display

get_ipython().run_line_magic('matplotlib', 'inline')


# The simplest form is to use `interact`, which we pass a function name and the variables with ranges.  By default this makes a *slider*, which takes on integer or floating point values depending on whether you put decimal points in the range. **Try it! Then modify the function and try again.**

# In[ ]:


# We can do this to any function
def test_f(x=5.):
    """Test function that prints the passed value and its square.
       Note that there is no return value in this case."""
    print ('x = ', x, ' and  x^2 = ', x**2)
    
widgets.interact(test_f, x=(0.,10.));


# In[ ]:


# Explicit declaration of the widget (here FloatSlider) and details
def test_f(x=5.):
    """Test function that prints the passed value and its square.
       Note that there is no return value in this case."""
    print ('x = ', x, ' and  x^2 = ', x**2)
    
widgets.interact(test_f, 
                 x = widgets.FloatSlider(min=-10,max=30,step=1,value=10));


# Here's an example with some bells and whistles for a plot.  **Try making changes!**

# In[ ]:


def plot_it(freq=1., color='blue', lw=2, grid=True, xlabel='x', 
            function='sin'):
    """ Make a simple plot of a trig function but allow the plot style
        to be changed as well as the function and frequency."""
    t = np.linspace(-1., +1., 1000)  # linspace(min, max, total #)

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(1,1,1)

    if function=='sin':
        ax.plot(t, np.sin(2*np.pi*freq*t), lw=lw, color=color)
    elif function=='cos':
        ax.plot(t, np.cos(2*np.pi*freq*t), lw=lw, color=color)
    elif function=='tan':
        ax.plot(t, np.tan(2*np.pi*freq*t), lw=lw, color=color)

    ax.grid(grid)
    ax.set_xlabel(xlabel)
    
widgets.interact(plot_it, 
                 freq=(0.1, 2.), color=['blue', 'red', 'green'], 
                 lw=(1, 10), xlabel=['x', 't', 'dog'],
                 function=['sin', 'cos', 'tan']);
    


# ### Further examples with numpy
# 
# * The [NumPy tutorial](https://www.numpy.org/devdocs/user/quickstart.html) is a good resource.
# * The [Datacamp NumPy tutorial](https://www.datacamp.com/community/tutorials/python-numpy-tutorial) covers much of the material in this notebook with more detail and also other topics.
# * Another tutorial is from [GeeksforGeeks](https://www.geeksforgeeks.org/numpy-tutorial/)
# 

# In[ ]:




