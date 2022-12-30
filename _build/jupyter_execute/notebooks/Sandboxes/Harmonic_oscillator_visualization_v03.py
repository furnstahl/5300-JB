#!/usr/bin/env python
# coding: utf-8

# # Harmonic oscillator visualization 
# 
# version 03.0: based on pendulum code; v2 adds driving force curve and changes $\theta$ to $q$; v3 adds the acceleration plot ($\ddot q$).
# 
# * Created 12-Jan-2019 by Dick Furnstahl (furnstahl.1@osu.edu)
# * Last revised 22-Jan-2021 by Dick Furnstahl (furnstahl.1@osu.edu).

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import numpy as np
from scipy.integrate import ode, odeint

import matplotlib.pyplot as plt


# ## Harmonic oscillator code

# In[3]:


class Harmonic_oscillator():
    """
    Harmonic oscillator class implements the parameters and differential 
     equation for a damped, driven, simple harmonic oscillator.
     
    Parameters
    ----------
    omega0 : float
        natural frequency of the oscillator (e.g., \sqrt{k/m} if a spring) 
    beta : float
        coefficient of damping term (with a factor of 2) 
    f_ext : float
        amplitude of external force (this is f_0 in Taylor) 
    omega_ext : float
        frequency of external force 
    phi_ext : float
        phase angle for external force (taken to be zero in Taylor)

    Methods
    -------
    dy_dt(y, t)
        Returns the right side of the differential equation in vector y, 
        given time t and the corresponding value of y.
        
    driving_force(t)
        Returns the external driving force at time t.
    """
    def __init__(self,
                 omega0=1.,
                 beta=0.2,
                 f_ext=0.2,
                 omega_ext=0.689,
                 phi_ext=0.
                ):
        self.omega0 = omega0
        self.beta = beta
        self.f_ext = f_ext
        self.omega_ext = omega_ext
        self.phi_ext = phi_ext
    
    def dy_dt(self, y, t):
        """
        This function returns the right-hand side of the diffeq as a vector: 
        [d q/dt 
         d^2q/dt^2]
        
        Parameters
        ----------
        y : float
            A 2-component vector with y[0] = q(t) and y[1] = dq/dt
        t : float
            time 
                    
        """
        F_ext = self.driving_force(t)
        return [y[1], -self.omega0**2*y[0] - 2.*self.beta*y[1] + F_ext]
    
    def driving_force(self, t):
        """
        This function returns the value of the driving force at time t.
        """
        return self.f_ext * np.cos(self.omega_ext*t + self.phi_ext)


# In[4]:


def plot_y_vs_x(x, y, axis_labels=None, label=None, title=None, 
                color=None, linestyle=None, ax=None):
    """
    Generic plotting function: return a figure axis with a plot of y vs. x,
    with line color and style, title, axis labels, and line label
    """
    if ax is None:        # if the axis object doesn't exist, make one
        ax = plt.gca()

    line, = ax.plot(x, y, label=label, color=color, linestyle=linestyle)
    if label is not None:    # if a label if passed, show the legend
        ax.legend()
    if title is not None:    # set a title if one if passed
        ax.set_title(title)
    if axis_labels is not None:  # set x-axis and y-axis labels if passed  
        ax.set_xlabel(axis_labels[0])
        ax.set_ylabel(axis_labels[1])

    return ax, line


# ## Interface using ipywidgets with interactive_output
# 
# We'll make a more elaborate interface so we can adjust all of the parameters.

# In[5]:


# Import the widgets we will use (add more if needed!) 
import ipywidgets as widgets
from ipywidgets import HBox, VBox, Layout, Tab, Label, Checkbox
from ipywidgets import FloatSlider, IntSlider, Play, Dropdown, HTMLMath 

from IPython.display import display
from time import sleep


# In[6]:


# This function generates the main output here, which is a grid of plots
def ho_plots(q_vs_time_plot=True, q_dot_vs_time_plot=True,
             q_ddot_vs_time_plot=True,
             phase_space_plot=True, driving_curve=True,
             omega0=10.*np.pi, beta=np.pi/2., 
             f_ext=1000., omega_ext=2.*np.pi, phi_ext=0., 
             q0=0.0, q_dot0=0.0,  
             t_start=0, t_end=10, delta_t=0.01, plot_start=0,
             font_size=18):
    """
    Create plots for interactive_output according to the inputs.
    
    Based on generating a Harmonic_oscillator instance and associated graphs.
    
    Notes
    -----
        1. We generate a new Harmonic_oscillator instance every time *and* 
            solve the ODE every time, even if the only change is to parameters
            like t_start and t_end.  Should we care or is this just so
            cheap to recalculate that it doesn't matter?
            How could we structure this differently?
        2. Should we delete ho1 at some point?  E.g., is there a memory issue?
    """
    
    # add delta_t it goes at least to t_end (probably should use linspace!)
    t_pts = np.arange(t_start, t_end+delta_t, delta_t)  
        
    # Instantiate an oscillator with the passed (or default) values of the 
    #  natural frequency omega0, damping beta, driving amplitude, frequency, 
    #  and phase (f_ext, omega_ext, phi_ext).
    ho1 = Harmonic_oscillator(omega0=omega0, beta=beta, f_ext=f_ext, 
                              omega_ext=omega_ext, phi_ext=phi_ext)
    y0 = [q0, q_dot0]  # initial conditions for the oscillator ODE

    # ODE solver parameters
    abserr = 1.0e-8
    relerr = 1.0e-6

    # For now we solve with odeint; give more options in the future.
    #  The .T is for transpose, so that the matrix from odeint can changed
    #  to the correct form for reading off q and q_dot.
    q, q_dot = odeint(ho1.dy_dt, y0, t_pts,
                              atol=abserr, rtol=relerr).T
    q_ddot = np.gradient(q_dot, t_pts)
    # also calculate the driving force for the same t_pts
    driving = ho1.driving_force(t_pts)
    
    # Update the common font size
    plt.rcParams.update({'font.size': font_size})
 
    # Labels for individual plot axes
    q_vs_time_labels = (r'$t$', r'$q$')
    q_dot_vs_time_labels = (r'$t$', r'$dq/dt$')
    q_ddot_vs_time_labels = (r'$t$', r'$d^2q/dt^2$')
    phase_space_labels = (r'$q$', r'$dq/dt$')
    
    # Figure out how many rows and columns [one row for now]
    plot_flags = [q_vs_time_plot, q_dot_vs_time_plot, q_ddot_vs_time_plot, phase_space_plot]
    plot_num = plot_flags.count(True)
    plot_rows = 1
    figsize_rows = plot_rows*6
    plot_cols = plot_num
    figsize_cols = min(plot_cols*8, 16)  # at most 16
    
    # Make the plot!
    fig = plt.figure(figsize=(figsize_cols,figsize_rows))
    #, axes = plt.subplots(plot_rows, plot_cols, 
                             
    # finds nearest index to plot_start in t_pts array                      
    start_index = (np.fabs(t_pts-plot_start)).argmin() 
    
    next_axis = 1  # keep track of the axis number
    if q_vs_time_plot:
        ax_q = fig.add_subplot(plot_rows, plot_cols, next_axis)                  
        plot_y_vs_x(t_pts, q, axis_labels=q_vs_time_labels, 
                    label='oscillator', title=r'$q$ vs. time', 
                    ax=ax_q)    
        # add a line where the phase space plot starts
        ax_q.axvline(t_pts[start_index], lw=3, color='red')
                          
        if driving_curve:
           ax_driving = ax_q.twinx()
           plot_y_vs_x(t_pts, driving, ax=ax_driving, color='red',
                       linestyle='dotted', label='driving')      # add 'driving label?'                   
        next_axis += 1
    
    if q_dot_vs_time_plot:
        ax_q_dot = fig.add_subplot(plot_rows, plot_cols, next_axis)                  
        plot_y_vs_x(t_pts, q_dot, axis_labels=q_dot_vs_time_labels, 
                    label='oscillator', title=r'$dq/dt$ vs. time', 
                    ax=ax_q_dot)    
        # add a line where the phase space plot starts
        ax_q_dot.axvline(t_pts[start_index], lw=3, color='red')
                          
        if driving_curve:
           ax_driving2 = ax_q_dot.twinx()
           plot_y_vs_x(t_pts, driving, ax=ax_driving2, color='red',
                       linestyle='dotted')  # add 'driving label?'                        
        next_axis += 1
    
    if q_ddot_vs_time_plot:
        ax_q_ddot = fig.add_subplot(plot_rows, plot_cols, next_axis)                  
        plot_y_vs_x(t_pts, q_ddot, axis_labels=q_ddot_vs_time_labels, 
                    label='oscillator', title=r'$d^2q/dt^2$ vs. time', 
                    ax=ax_q_ddot)    
        # add a line where the phase space plot starts
        ax_q_ddot.axvline(t_pts[start_index], lw=3, color='red')
                          
        if driving_curve:
           ax_driving3 = ax_q_ddot.twinx()
           plot_y_vs_x(t_pts, driving, ax=ax_driving3, color='red',
                       linestyle='dotted')  # add 'driving label?'                        
        next_axis += 1

    if phase_space_plot:
        ax_phase_space = fig.add_subplot(plot_rows, plot_cols, next_axis)                  
        plot_y_vs_x(q[start_index:-1], q_dot[start_index:-1], 
                    axis_labels=phase_space_labels, title='State space', 
                    ax=ax_phase_space)    
        next_axis += 1
    
    fig.tight_layout()
    
    return fig



# In[7]:


# Widgets for the various inputs.
#   For any widget, we can set continuous_update=False if we don't want the 
#    plots to shift until the selection is finished (particularly relevant for 
#    sliders).

# Widgets for the plot choice (plus a label out front)
plot_choice_w = Label(value='Which plots: ',layout=Layout(width='100px'))
def plot_choice_widget(on=True, plot_description=None):
    """Makes a Checkbox to select whether to show a plot."""
    return Checkbox(value=on, description=plot_description,
                  disabled=False, indent=False, layout=Layout(width='150px'))
q_vs_time_plot_w = plot_choice_widget(True, r'$q$ vs. time')
q_dot_vs_time_plot_w = plot_choice_widget(False, r'$dq/dt$ vs. time')
q_ddot_vs_time_plot_w = plot_choice_widget(False, r'$d^2q/dt^2$ vs. time')
phase_space_plot_w = plot_choice_widget(True, 'state space')
driving_curve_w = plot_choice_widget(False, 'driving force')


# Widgets for the oscillator parameters (all use FloatSlider, so we made 
#  it a function)
def float_widget(value, min, max, step, description, format):
    """Makes a FloatSlider with the passed parameters and continuous_update
       set to False."""
    slider_border = Layout(border='solid 1.0px')
    return FloatSlider(value=value,min=min,max=max,step=step,disabled=False,
                       description=description,continuous_update=False,
                       orientation='horizontal',layout=slider_border,
                       readout=True,readout_format=format)

omega0_w = float_widget(value=10.*np.pi, min=0.0, max=20.*np.pi, step=0.1,
                        description=r'natural $\omega_0$:', format='.2f')
beta_w = float_widget(value=np.pi/2., min=0.0, max=2.*np.pi, step=0.1,
                       description=r'damping $\beta$:', format='.2f')
f_ext_w = float_widget(value=1000., min=0.0, max=2000., step=1.,
                       description=r'drive $f_{\rm ext}$:', format='.0f')
omega_ext_w = float_widget(value=2.*np.pi, min=0.0, max=6.*np.pi, step=0.1,
                       description=r'freq. $\omega_{\rm ext}$:', format='.2f')
phi_ext_w = float_widget(value=0.0, min=0, max=2.*np.pi, step=0.1,
                         description=r'phase $\phi_{\rm ext}$:', format='.1f')

# Widgets for the initial conditions
q0_w = float_widget(value=0.0, min=-2.*np.pi, max=2.*np.pi, step=0.1,
                        description=r'$q_0$:', format='.1f')
q_dot0_w = float_widget(value=0.0, min=-100., max=100., step=1.,
                            description=r'$(dq/dt)_0$:', format='.1f')

# Widgets for the plotting parameters
t_start_w = float_widget(value=0., min=0., max=10., step=1.,
                         description='t start:', format='.1f') 
t_end_w = float_widget(value=5., min=0., max=20., step=1.,
                       description='t end:', format='.1f')
delta_t_w = float_widget(value=0.001, min=0.001, max=0.1, step=0.001,
                         description='delta t:', format='.3f')
plot_start_w = float_widget(value=0., min=0., max=20., step=1.,
                            description='start plotting:', format='.1f')

# Widgets for the styling parameters
font_size_w = Dropdown(options=['12', '16', '18', '20', '24'], value='18',
                       description='Font size:',disabled=False,
                       continuous_update=False,layout=Layout(width='140px'))


############## Begin: Explicit callback functions #######################

# Make sure that t_end is at least t_start + 10
def update_t_end(*args):
    if t_end_w.value < t_start_w.value:
        t_end_w.value = t_start_w.value + 10     
t_end_w.observe(update_t_end, 'value')
t_start_w.observe(update_t_end, 'value')


# Make sure that plot_start is at least t_start and less than t_end
def update_plot_start(*args):
    if plot_start_w.value < t_start_w.value:
        plot_start_w.value = t_start_w.value
    if plot_start_w.value > t_end_w.value:
        plot_start_w.value = t_end_w.value
plot_start_w.observe(update_plot_start, 'value')
t_start_w.observe(update_plot_start, 'value')
t_end_w.observe(update_plot_start, 'value')


############## End: Explicit callback functions #######################

# Set up the interactive_output widget 
plot_out = widgets.interactive_output(ho_plots,
                          dict(
                          q_vs_time_plot=q_vs_time_plot_w,
                          q_dot_vs_time_plot=q_dot_vs_time_plot_w,
                          q_ddot_vs_time_plot=q_ddot_vs_time_plot_w,
                          phase_space_plot=phase_space_plot_w,
                          driving_curve = driving_curve_w,
                          omega0=omega0_w,
                          beta=beta_w,
                          f_ext=f_ext_w,
                          omega_ext=omega_ext_w,
                          phi_ext=phi_ext_w,
                          q0=q0_w,
                          q_dot0=q_dot0_w,
                          t_start=t_start_w,
                          t_end=t_end_w, 
                          delta_t=delta_t_w,    
                          plot_start=plot_start_w, 
                          font_size=font_size_w)
                       )

# Now do some manual layout, where we can put the plot anywhere using plot_out
hbox1 = HBox([plot_choice_w, q_vs_time_plot_w, q_dot_vs_time_plot_w, q_ddot_vs_time_plot_w,
              phase_space_plot_w, driving_curve_w]) #  choice of plots to show
hbox2 = HBox([omega0_w, f_ext_w, omega_ext_w, phi_ext_w]) # external driving parameters
hbox3 = HBox([q0_w, q_dot0_w, beta_w]) # initial conditions and damping
hbox4 = HBox([t_start_w, t_end_w, delta_t_w, plot_start_w]) # time, plot ranges
hbox5 = HBox([font_size_w]) # font size

# We'll set up Tabs to organize the controls.  The Tab contents are declared
#  as tab0, tab1, ... (probably should make this a list?) and the overall Tab
#  is called tab (so its children are tab0, tab1, ...).
tab_height = '40px'  # Fixed minimum height for all tabs. Specify another way?
tab0 = VBox([hbox2, hbox3], layout=Layout(min_height=tab_height))
tab1 = VBox([hbox1, hbox4], layout=Layout(min_height=tab_height))
tab2 = VBox([hbox5], layout=Layout(min_height=tab_height))

tab = Tab(children=[tab0, tab1, tab2])
tab.set_title(0, 'Physics')
tab.set_title(1, 'Plotting')
tab.set_title(2, 'Styling')

# Release the Kraken!
vbox2 = VBox([tab, plot_out])
display(vbox2)


# In[ ]:





# In[ ]:




