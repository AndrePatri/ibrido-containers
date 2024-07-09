from isaacsim import SimulationApp

# The most basic usage for creating a simulation app
kit = SimulationApp(experience="/isaac-sim/apps/omni.isaac.sim.python.omnirobogym.headless.kit")

import omni

for i in range(100):
    kit.update()

omni.kit.app.get_app().print_and_log("Shader cache warmed up!")

kit.close()  # Cleanup application
