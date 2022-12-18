

from inputs import Inputs
from multiprocessing import Pool
import numpy as np
import os
import sys
sys.path.append(os.path.join('..','contrib','RocketPy'))
from rocketpy import Environment, SolidMotor, Rocket, Flight

import time

class Simulation:

    def __init__(self, heading=0):

        self.env = Environment(railLength=5.2)
        self.env.setAtmosphericModel(type='StandardAtmosphere')

        self.heading = heading



    def make_motor(self, input_data):

        return SolidMotor(
        thrustSource=input_data[0],
        #thrustSource="data/motors/custom.eng",
        burnOut=input_data[1],
        grainNumber=int(input_data[2]),
        grainSeparation=input_data[3],
        grainDensity=input_data[4],
        grainOuterRadius=33/1000,
        grainInitialInnerRadius=15/1000,
        grainInitialHeight=120/1000,
        nozzleRadius=input_data[5],
        throatRadius=input_data[6],
        interpolationMethod='linear'
        ) 


    def make_rocket(self, input_data):

        motor = self.make_motor(input_data)

        Calisto = Rocket(
        motor=motor,
        radius=input_data[7],
        mass=input_data[8],
        inertiaI=input_data[9],
        inertiaZ=input_data[10],
        distanceRocketNozzle=input_data[11],
        distanceRocketPropellant=input_data[12],
        powerOffDrag='../contrib/RocketPy/data/calisto/powerOffDragCurve.csv',
        powerOnDrag='../contrib/RocketPy/data/calisto/powerOnDragCurve.csv'
        )

    
        Calisto.setRailButtons([0.2, -0.5])

        nose_kind = 'conical'

        NoseCone = Calisto.addNose(length=input_data[13], kind=nose_kind, distanceToCM=input_data[14])
             
            
        FinSet = Calisto.addFins(int(input_data[15]), span=input_data[16], rootChord=input_data[17], tipChord=input_data[18], distanceToCM=input_data[19])

        Tail = Calisto.addTail(topRadius=input_data[20], bottomRadius=input_data[21], length=input_data[22], distanceToCM=input_data[23])

        return Calisto

    def run(self, input_data):

        rocket = self.make_rocket(input_data)
       
        flight = Flight(rocket=rocket, environment=self.env, inclination=input_data[24], heading=input_data[25])

        return flight

def main(input_tuple):

    i, (input_data, normed_data) = input_tuple


    try:
        flight = sim.run(input_data)
      
        traj = np.array(flight.solution)

        np.savetxt(os.path.join(out_dir, 'Outputs_run_{:07d}.txt'.format(i)),traj)
        np.savetxt(os.path.join(out_dir, 'Inputs_run_{:07d}.txt'.format(i)), np.array(input_data))
        np.savetxt(os.path.join(out_dir, 'NormInputs_run_{:07d}.txt'.format(i)), normed_data)


    except ValueError:
        pass

if __name__ == '__main__':


    mc_yaml = os.path.join('..','inputs','mc_params.yaml')
    out_dir = os.path.join('..','outputs','Raw','Test')

    

    n = 5000
    n_cores = 12


    # make output folder
    os.makedirs(out_dir,exist_ok=True)

    # Draw inputs  
    inputs = Inputs(mc_yaml)
    inputs.draw(n=n)

    chunksize = int(n / n_cores)

    sim = Simulation()


    iterable = enumerate(inputs)

    start = time.time()
    with Pool(processes=n_cores) as pool:

        pool.map(main, iterable, chunksize=chunksize)

    end = time.time()

    print('Time: {:3.4f}'.format(end-start))
