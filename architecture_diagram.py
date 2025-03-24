from diagrams import Diagram, Cluster, Edge
from diagrams.programming.language import Python
from diagrams.onprem.compute import Server
from diagrams.generic.compute import Rack
from diagrams.programming.framework import Django

# Create architecture diagram for the original Python implementation
with Diagram("Kp+FASTSIM+USFD Python Implementation Architecture", show=False, direction="TB", 
             filename="architecture_diagram"):
    
    # Input Data
    with Cluster("Input Data"):
        wheel_profile = Server("Wheel Profile")
        rail_profile = Server("Rail Profile")
        material_props = Server("Material Properties")
        creepage = Server("Creepage Parameters")
        
    # Individual Components
    with Cluster("Component Models"):
        # Kp Model
        with Cluster("Kp Contact Model"):
            kp_model = Python("kp_model.py")
            contact_patch = Rack("Contact Patch")
            pressure = Rack("Pressure Distribution")
            
            kp_model >> Edge(label="calculates") >> contact_patch
            kp_model >> Edge(label="calculates") >> pressure
        
        # FASTSIM Model
        with Cluster("FASTSIM Algorithm"):
            fastsim = Python("fastsim.py")
            tangential_stress = Rack("Tangential Stress")
            creep_forces = Rack("Creep Forces")
            
            fastsim >> Edge(label="calculates") >> tangential_stress
            fastsim >> Edge(label="calculates") >> creep_forces
        
        # USFD Model
        with Cluster("USFD Wear Model"):
            usfd_model = Python("usfd_wear_model.py")
            t_gamma = Rack("T-gamma")
            wear_rate = Rack("Wear Rate")
            
            usfd_model >> Edge(label="calculates") >> t_gamma
            usfd_model >> Edge(label="calculates") >> wear_rate
    
    # Integration Module
    with Cluster("Integration"):
        integration = Python("kp_fastsim_usfd_integration.py")
        visualization = Python("matplotlib visualization")
        
        integration >> Edge(label="provides") >> visualization
    
    # Data Flow
    wheel_profile >> Edge(label="input") >> kp_model
    rail_profile >> Edge(label="input") >> kp_model
    material_props >> Edge(label="input") >> kp_model
    material_props >> Edge(label="input") >> fastsim
    creepage >> Edge(label="input") >> fastsim
    
    contact_patch >> Edge(label="input") >> fastsim
    pressure >> Edge(label="input") >> fastsim
    
    creep_forces >> Edge(label="input") >> usfd_model
    creepage >> Edge(label="input") >> usfd_model
    
    # Integration connections
    kp_model >> Edge(label="component") >> integration
    fastsim >> Edge(label="component") >> integration
    usfd_model >> Edge(label="component") >> integration
