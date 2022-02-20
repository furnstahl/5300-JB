#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from math import factorial

# We'll get our uniform distributions from stats, but there are other ways.
import scipy.stats as stats  

import matplotlib.pyplot as plt
import seaborn; seaborn.set() # for nicer plot formatting


# In[10]:


x = (1,2,3,4)
AMO_label = "AMO"
AMO_votes = (5,0,3,1)
Astro_label = "Astro"
Astro_votes = (2,5,1,2)


# In[12]:


fig_1 = plt.figure(figsize=(10,5))
ax_1 = fig_1.add_subplot(1,1,1)
ax_1.bar(x, AMO_votes, color="blue", label=AMO_label)
ax_1.bar(x, Astro_votes, color="red", label=Astro_label)
ax_1.legend();

fig_1.tight_layout()


# In[ ]:




