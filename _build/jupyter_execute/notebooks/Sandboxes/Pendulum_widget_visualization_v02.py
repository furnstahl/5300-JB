#!/usr/bin/env python
# coding: utf-8

# # Pendulum visualization using ipywidgets
# 
# v2 adds driving force curve
# 
# * Created 12-Dec-2018 by Dick Furnstahl (furnstahl.1@osu.edu)
# * Last revised 19-Jan-2019 by Dick Furnstahl (furnstahl.1@osu.edu).

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import numpy as np
from scipy.integrate import ode, odeint

import matplotlib.pyplot as plt


# ## Pendulum code

# In[ ]:


class Pendulum():   
    """
    Pendulum class implements the parameters and differential equation for 
     a pendulum using the notation from Taylor.
     
    Parameters
    ----------
    omega_0 : float
        natural frequency of the pendulum (\sqrt{g/l} where l is the 
        pendulum length) 
    beta : float
        coefficient of friction 
    gamma_ext : float
        amplitude of external force is gamma * omega_0**2 
    omega_ext : float
        frequency of external force 
    phi_ext : float
        phase angle for external force 

    Methods
    -------
    dy_dt(y, t)
        Returns the right side of the differential equation in vector y, 
        given time t and the corresponding value of y.
    """
    def __init__(self,
                 omega_0=1.,
                 beta=0.2,
                 gamma_ext=0.2,
                 omega_ext=0.689,
                 phi_ext=0.
                ):
        self.omega_0 = omega_0
        self.beta = beta
        self.gamma_ext = gamma_ext
        self.omega_ext = omega_ext
        self.phi_ext = phi_ext
    
    def dy_dt(self, y, t):
        """
        This function returns the right-hand side of the diffeq: 
        [dphi/dt d^2phi/dt^2]
        
        Parameters
        ----------
        y : float
            A 2-component vector with y[0] = phi(t) and y[1] = dphi/dt
        t : float
            time 
            
        Returns
        -------
        
        """
        F_ext = self.driving_force(t)
        return [y[1], -self.omega_0**2 * np.sin(y[0]) - 2.*self.beta * y[1]                        + F_ext]
    
    def driving_force(self, t):
        """
        This function returns the value of the driving force at time t.
        """
        return self.gamma_ext * self.omega_0**2                               * np.cos(self.omega_ext*t + self.phi_ext)   


# In[ ]:


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

# In[ ]:


# Import the widgets we will use (add more as needed!) 
import ipywidgets as widgets
from ipywidgets import HBox, VBox, Layout, Tab, Label, Checkbox
from ipywidgets import FloatSlider, IntSlider, Play, Dropdown, HTMLMath 

from IPython.display import display
from time import sleep


# In[ ]:


# This function generates the main output, which is a grid of plots
def pendulum_plots(phi_vs_time_plot=True, phi_dot_vs_time_plot=True, 
                   state_space_plot=True, driving_curve=True,
                   omega_0=1., beta=0.2, 
                   gamma_ext=0.2, omega_ext=0.689, phi_ext=0., 
                   phi_0=0.8, phi_dot_0=0.0, 
                   t_start=0, t_end=100, delta_t=0.1, plot_start=0,
                   font_size=18):
    """
    Create plots for interactive_output according to the inputs.
    
    Based on generating a Pendulum instance and the requested graphs.
    
    Notes
    -----
        1. We generate a new Pendulum instance every time *and* solved
            the ODE every time, even if the only change is to parameters
            like t_start and t_end.  Should we care or is this just so
            cheap to recalculate that it doesn't matter?
            How could we structure this differently?
    """
    
    # add delta_t o it goes at least to t_end (probably should use linspace)
    t_pts = np.arange(t_start, t_end+delta_t, delta_t)  
        
    # Instantiate a pendulum with the passed (or default) values of the 
    #  natural frequency omega_0, damping beta, driving amplitude, frequency, 
    #  and phase (f_ext, omega_ext, phi_ext).
    #  Should we delete p1 at some point?  Is there a memory issue?
    p1 = Pendulum(omega_0=omega_0, beta=beta, 
                  gamma_ext=gamma_ext, omega_ext=omega_ext, phi_ext=phi_ext)
    y0 = [phi_0, phi_dot_0]  # initial conditions for the pendulum ODE

    # ODE solver parameters
    abserr = 1.0e-8
    relerr = 1.0e-6

    # For now we solve with odeint; give more options in the future.
    phi, phi_dot = odeint(p1.dy_dt, y0, t_pts, atol=abserr, rtol=relerr).T
    
    # also calculate the driving force for the same t_pts
    driving = p1.driving_force(t_pts)
    
    # Update the common font size
    plt.rcParams.update({'font.size': font_size})
 
    # Labels for individual plot axes
    phi_vs_time_labels = (r'$t$', r'$\phi$')
    phi_dot_vs_time_labels = (r'$t$', r'$d\phi/dt$')
    state_space_labels = (r'$\phi$', r'$d\phi/dt$')
    
    # Figure out how many rows and columns [one row for now]
    plot_flags = [phi_vs_time_plot, phi_dot_vs_time_plot, state_space_plot]
    plot_num = plot_flags.count(True)
    plot_rows = 1
    figsize_rows = plot_rows*6
    plot_cols = plot_num
    figsize_cols = min(plot_cols*8, 16)  # at most 16

    # Make the plot!
    fig = plt.figure(figsize=(figsize_cols,figsize_rows))
                             
    # finds nearest index to plot_start in t_pts array                      
    start_index = (np.fabs(t_pts-plot_start)).argmin() 
    
    next_axis = 1  # keep track of the axis number
    if phi_vs_time_plot:
        ax_phi = fig.add_subplot(plot_rows, plot_cols, next_axis)                  
        plot_y_vs_x(t_pts, phi, axis_labels=phi_vs_time_labels, 
                    label='pendulum', title=r'$\phi$ vs. time', 
                    ax=ax_phi)    
        # add a line where the state space plot starts
        ax_phi.axvline(t_pts[start_index], lw=3, color='red')
                          
        if driving_curve:
           ax_driving = ax_phi.twinx()
           plot_y_vs_x(t_pts, driving, ax=ax_driving, color='red',
                       linestyle='dotted')      # add 'driving label?'                   
        next_axis += 1
    
    if phi_dot_vs_time_plot:
        ax_phi_dot = fig.add_subplot(plot_rows, plot_cols, next_axis)                  
        plot_y_vs_x(t_pts, phi_dot, axis_labels=phi_dot_vs_time_labels, 
                    label='oscillator', title=r'$dq/dt$ vs. time', 
                    ax=ax_phi_dot)    
        # add a line where the phase space plot starts
        ax_phi_dot.axvline(t_pts[start_index], lw=3, color='red')
                          
        if driving_curve:
           ax_driving2 = ax_phi_dot.twinx()
           plot_y_vs_x(t_pts, driving, ax=ax_driving2, color='red',
                       linestyle='dotted')  # add 'driving label?'                        
        next_axis += 1

    if state_space_plot:
        ax_state_space = fig.add_subplot(plot_rows, plot_cols, next_axis)                  
        plot_y_vs_x(phi[start_index:-1], phi_dot[start_index:-1], 
                    axis_labels=state_space_labels, title='State space', 
                    ax=ax_state_space)    
        next_axis += 1
    
    fig.tight_layout()
    
    return fig


# In[ ]:


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
phi_vs_time_plot_w = plot_choice_widget(True, r'$\phi$ vs. time')
phi_dot_vs_time_plot_w = plot_choice_widget(False, r'$d\phi/dt$ vs. time')
state_space_plot_w = plot_choice_widget(True, 'state space')
driving_curve_w = plot_choice_widget(True, 'driving force')

# Widgets for the pendulum parameters (all use FloatSlider, so made function)
def float_widget(value, min, max, step, description, format):
    """Makes a FloatSlider with the passed parameters and continuous_update
       set to False."""
    slider_border = Layout(border='solid 1.0px')
    return FloatSlider(value=value,min=min,max=max,step=step,disabled=False,
                       description=description,continuous_update=False,
                       orientation='horizontal',layout=slider_border,
                       readout=True,readout_format=format)

omega_0_w = float_widget(value=1.0, min=0.0, max=10., step=0.1,
                         description=r'natural $\omega_0$:', format='.1f')
beta_w = float_widget(value=0.1, min=0.0, max=2., step=0.1,
                      description=r'damping $\beta$:', format='.1f')
gamma_ext_w = float_widget(value=0.2, min=0.0, max=2., step=0.05,
                       description=r'driving $\gamma$:', format='.2f')
omega_ext_w = float_widget(value=0.689,min=0.0,max=3.,step=0.1,
                       description=r'freq. $\omega_{\rm ext}$:', format='.2f')
phi_ext_w = float_widget(value=0.0, min=0, max=2.*np.pi, step=0.1,
                         description=r'phase $\phi_{\rm ext}$:', format='.1f')

# Widgets for the initial conditions
phi_0_w = float_widget(value=0.8, min=0., max=2.*np.pi, step=0.1,
                        description=r'$\phi_0$:', format='.1f')
phi_dot_0_w = float_widget(value=0.0, min=-10., max=10., step=0.1,
                            description=r'$(d\phi/dt)_0$:', format='.1f')

# Widgets for the plotting parameters
t_start_w = float_widget(value=0., min=0., max=100., step=10.,
                         description='t start:', format='.1f') 
t_end_w = float_widget(value=100., min=0., max=500., step=10.,
                       description='t end:', format='.1f')
delta_t_w = float_widget(value=0.1, min=0.01, max=0.2, step=0.01,
                         description='delta t:', format='.2f')
plot_start_w = float_widget(value=0., min=0., max=300., step=5.,
                            description='start plotting:', format='.1f')

# Widgets for the styling parameters
font_size_w = Dropdown(options=['12', '16', '18', '20', '24'], value='18',
                       description='Font size:',disabled=False,
                       continuous_update=False,layout=Layout(width='140px'))


############## Begin: Explicit callback functions #######################

# Make sure that t_end is at least t_start + 50
def update_t_end(*args):
    if t_end_w.value < t_start_w.value:
        t_end_w.value = t_start_w.value + 50     
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
plot_out = widgets.interactive_output(pendulum_plots,
                          dict(
                          phi_vs_time_plot=phi_vs_time_plot_w,
                          phi_dot_vs_time_plot=phi_dot_vs_time_plot_w,
                          state_space_plot=state_space_plot_w,
                          driving_curve = driving_curve_w,
                          omega_0=omega_0_w,
                          beta=beta_w,
                          gamma_ext=gamma_ext_w,
                          omega_ext=omega_ext_w,
                          phi_ext=phi_ext_w,
                          phi_0=phi_0_w,
                          phi_dot_0=phi_dot_0_w,
                          t_start=t_start_w,
                          t_end=t_end_w, 
                          delta_t=delta_t_w,    
                          plot_start=plot_start_w, 
                          font_size=font_size_w)
                       )

# Now do some manual layout, where we can put the plot anywhere using plot_out
hbox1 = HBox([plot_choice_w, phi_vs_time_plot_w, phi_dot_vs_time_plot_w,
              state_space_plot_w, driving_curve_w]) #  choice of what plots 
hbox2 = HBox([omega_0_w, gamma_ext_w, omega_ext_w, phi_ext_w])  # external 
                                                         # driving parameters
hbox3 = HBox([phi_0_w, phi_dot_0_w, beta_w]) # initial conditions and damping
hbox4 = HBox([t_start_w, t_end_w, delta_t_w, plot_start_w]) # time and plot ranges
hbox5 = HBox([font_size_w]) # font size

# We'll set up Tabs to organize the controls.  The Tab contents are declared
#  as tab0, tab1, ... (probably should make this a list?) and the overall Tab
#  is called tab (so its children are tab0, tab1, ...).
tab_height = '70px'  # Fixed minimum height for all tabs. Specify another way?
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




