"""
 Title:         EVP Model
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
        self.define_param("n",   1.0e0, 1.0e2)
        self.define_param("eta", 0.0e1, 1.0e4)
        self.define_param("s0",  2.0e2, 5.0e3)
        self.define_param("R",   0.0e0, 1.0e4)
        self.define_param("d",   0.0e1, 1.0e3)
        self.youngs = self.get_constant(211000)
    
    # Returns the model integrator
    def get_integrator(self, n, eta, s0, R, d, **kwargs):
        isotropic  = hardening.VoceIsotropicHardeningModel(R, d)
        kinematic  = hardening.NoKinematicHardeningModel()
        flowrule   = flowrules.IsoKinViscoplasticity(n, eta, s0, isotropic, kinematic)
        evp_model  = models.InelasticModel(self.youngs, flowrule)
        integrator = models.ModelIntegrator(evp_model, **kwargs) 
        return integrator
