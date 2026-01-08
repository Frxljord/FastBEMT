import pickle
from src.Propeller import Propeller
from src.JobParameters import JobParameters

def main():
    with open("10x7E.pkl", "rb") as f:
        bladeDict = pickle.load(f)

    propRadius = bladeDict['tip_radius']
    hubRadius = bladeDict['hub_radius']
    nBlades = bladeDict['n_blades']

    # Operating conditions
    RPM = 7000
    vInf = 0

    # Fluid Parameters
    rho = 1.225
    mu = 1.81e-5
    aInf = 343

    # Define propeller parameters
    propellerParams = JobParameters(
        propRadius=propRadius,
        hubRadius=hubRadius,
        nBlades=nBlades,
        RPM=RPM,
        vInf=vInf,
        aInf=aInf,
        rho=rho,
        mu=mu
    )

    analysis = Propeller(
        propellerGeometry=bladeDict,
        propellerParams=propellerParams
    )
    analysis.runBEMT()

    thrust, torque, Ct, Cp = analysis.computeTotalForces()
    print(f"Last run totals: thrust={thrust:.4f} N, torque={torque:.4f} N·m, Ct={Ct:.6f}, Cp={Cp:.6f}")


if __name__ == "__main__":
    main()
