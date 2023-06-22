"""
 Title:         EVP Test Model (provided in PyOptMat)
 Description:   Elastic Viscoplastic Model
 Author:        Janzen Choi

"""

# Libraries
from myoptmat.models.__model__ import __Model__
from pyoptmat import models, flowrules, hardening

# Model class
class Model(__Model__):
    
    # Sets up the model
    def prepare(self):
        self.define_param("n",   7*0.5,   7*1.5)
        self.define_param("eta", 300*0.5, 300*1.5)
        self.define_param("s0",  50*0.5,  50*1.5)
        self.define_param("R",   200*0.5, 200*1.5)
        self.define_param("d",   5*0.5,   5*1.5)
        self.youngs = self.get_constant(150000)
    
    # Returns the model integrator
    def get_integrator(self, n, eta, s0, R, d, **kwargs):
        isotropic  = hardening.VoceIsotropicHardeningModel(R, d)
        kinematic  = hardening.NoKinematicHardeningModel()
        flowrule   = flowrules.IsoKinViscoplasticity(n, eta, s0, isotropic, kinematic)
        evp_model  = models.InelasticModel(self.youngs, flowrule)
        integrator = models.ModelIntegrator(evp_model, **kwargs) 
        return integrator
