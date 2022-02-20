#!/usr/bin/env python
# coding: utf-8

# # Non-physics example of using Python subclasses
# 
# Adapted from http://www.jesshamrick.com/2011/05/18/an-introduction-to-classes-and-inheritance-in-python/.

# In[ ]:


class Pet:
    """Base or parent class for generic pet methods"""
    def __init__(self, name, species):
        self.name = name
        self.species = species

    def getName(self):
        return self.name

    def getSpecies(self):
        return self.species

    def __str__(self):
        return "%s is a %s" % (self.name, self.species)

class Dog(Pet):
    """Subclass of Pet for dogs with specialized methods"""
    def __init__(self, name, chases_cats):
        Pet.__init__(self, name, "Dog")
        self.chases_cats = chases_cats

    def chasesCats(self):
        return self.chases_cats

class Cat(Pet):
    """Subclass of Pet for cats with specialized methods"""
    def __init__(self, name, hates_dogs):
        Pet.__init__(self, name, "Cat")
        self.hates_dogs = hates_dogs

    def hatesDogs(self):
        return self.hates_dogs


# ## Let's give the Pet class a test drive

# In[ ]:


polly = Pet("Polly", "Parrot")
print(f'The name of the pet is {polly.name}.')
print(f'The species of the pet is {polly.species}.')
print(polly)


# ## Now try the Cat subclass

# In[ ]:


ginger = Cat("Ginger", True)
print(f'The name of the pet is {ginger.name}.')
print(f'The species of the pet is {ginger.species}.')
print(ginger)
print(f'{ginger.name} hates dogs: {ginger.hatesDogs()}')


# ## Try these!

# In[ ]:


fido = Dog("Fido", True)
rover = Dog("Rover", False)
mittens = Cat("Mittens", True)
fluffy = Cat("Fluffy", False)


# In[ ]:




