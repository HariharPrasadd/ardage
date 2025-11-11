##### EUROPEAN ORGANISATION FOR NUCLEAR RESEARCH (CERN)



Comput Softw Big Sci 6 (2022) 7
[DOI: 10.1007/s41781-021-00079-7](http://dx.doi.org/10.1007/s41781-021-00079-7)



CERN-EP-2021-174

November 25, 2024


# **AtlFast3: the next generation of fast simulation in** **ATLAS**

#### The ATLAS Collaboration

The ATLAS experiment at the Large Hadron Collider has a broad physics programme ranging
from precision measurements to direct searches for new particles and new interactions,
requiring ever larger and ever more accurate datasets of simulated Monte Carlo events.
Detector simulation with Geant4 is accurate but requires significant CPU resources. Over the
past decade, ATLAS has developed and utilized tools that replace the most CPU-intensive
component of the simulation – the calorimeter shower simulation – with faster simulation
methods. Here, AtlFast3, the next generation of high-accuracy fast simulation in ATLAS, is
introduced. AtlFast3 combines parameterized approaches with machine-learning techniques
and is deployed to meet current and future computing challenges and simulation needs of the
ATLAS experiment. With highly accurate performance and significantly improved modelling
of substructure within jets, AtlFast3 can simulate large numbers of events for a wide range of
physics processes.


© 2024 CERN for the benefit of the ATLAS Collaboration.

Reproduction of this article or parts of it is allowed as specified in the CC-BY-4.0 license.


#### **Contents**

**1** **Introduction** **2**


**2** **The ATLAS detector** **4**

2.1 Inner detector 5

2.2 Calorimeters 5

2.3 Muon spectrometer 6


**3** **Dataset simulation and preprocessing** **7**
3.1 Simulation of reference samples with Geant4 7
3.2 Voxelization 8

3.3 Validation datasets 10


**4** **Calorimeter simulation with FastCaloSim V2** **11**

4.1 Longitudinal shower development 11
4.2 Average lateral shower shape 14
4.3 Simulation of hits 15


**5** **Calorimeter simulation with FastCaloGAN** **20**

5.1 Architecture 20

5.2 Training 21
5.3 Simulation of hits 23


**6** **Simulation of muon punch-through** **28**


**7** **The combination of FastCaloSim V2 and FastCaloGAN: AtlFast3** **29**

7.1 Configuration of AtlFast3 29
7.2 Configuration of the Fast Calorimeter Simulation 31
7.3 Energy interpolation 34
7.4 Corrections 35


**8** **Performance of AtlFast3** **40**

8.1 Performance of AtlFast3 on objects for physics analysis 40
8.2 Performance of AtlFast3 in physics analysis 48
8.3 Computing performance with AtlFast3 52


**9** **Conclusion** **53**

#### **1 Introduction**


The physics programme of the ATLAS experiment [1] at the Large Hadron Collider [2] (LHC) relies on the
accurate simulation of billions of Monte Carlo (MC) events to complement the data delivered by the LHC.
Simulated events are prepared by generating the appropriate physics process, simulating the passage of
particles through the detectors, digitizing the detector response, and then reconstructing those events using
the same reconstruction algorithms that are applied to the recorded data. The simulation of the passage of


2


particles through the detectors for analyses of data taken during LHC Run 2 requires approximately 40% of
the computing resources of the ATLAS experiment [3, 4]. Run 2 was the second data-taking run of the
LHC and lasted from 2015 to 2018.


The complex accordion geometry of the ATLAS electromagnetic calorimeter (see Section 2.2) makes the
simulation of shower development particularly CPU intensive when using the Geant4 [5] toolkit (G4).
In fact, 80% of the total simulation time for a typical sample of the production of top and anti-top quark
pair ( _𝑡𝑡_ ¯ ) is devoted to modelling this shower development [6]. Therefore, fast approaches for calorimeter
simulation are crucial in reducing the CPU needs of the ATLAS experiment and to enable the production
of the required numbers of simulated events for precision physics analyses.


The ATLAS Collaboration has already developed and deployed a fast simulation tool, called AtlFastII [6]
(AF2), that performs the simulation of the entire ATLAS detector by combining different tools to simulate
different sub-detectors and particles. In particular, AtlFastII relies on a fast simulation of the calorimeters
called FastCaloSim [7]. AtlFastII has been used to produce approximately 32 billion of the 52 billion
events simulated for physics analyses of the Run 2 data. AtlFastII is known to have certain limitations,
particularly in the modelling of jets of particles reconstructed with large-radius clustering algorithms and
the detailed description of their substructure. In this paper, the ATLAS Collaboration introduces a new fast
simulation tool, AtlFast3 or AF3, which has the same CPU performance as AtlFastII, but better accuracy in
reproducing Geant4. The ATLAS Collaboration is using AtlFast3 for a large resimulation campaign of
Run 2 MC events and plans to use AtlFast3 extensively for Run 3 and beyond.


AtlFastII and AtlFast3 perform the simulation of the entire ATLAS detector by combining a number of
components. Key components of AtlFast3 are two parametric calorimeter simulations: the new version of
FastCaloSim, referred to as FastCaloSim V2, and FastCaloGAN. Parametric simulations of the calorimeter
response simulate the energy of a particle shower as a single step based on an underlying parametrization
instead of simulating how every particle propagates and interacts inside the calorimeter volume. This
makes the CPU performance of these tools essentially independent of the particle energy and scales linearly
with the number of particles entering the calorimeter volume.


AtlFast3 is designed to simulate particle showers to a level of precision such that no sizeable differences
from the Geant4 simulation can be resolved by the reconstruction algorithms, including those for electron,
photon, and _𝜏_ -lepton reconstruction and identification, and jet reconstruction and clustering.


The fast simulation of showers in the calorimeter can be factorized into several components: the total
shower energy, the energy sharing between calorimeter layers, the average lateral shower development
within a layer, the uncorrelated energy fluctuations in individual showers compared to average showers and,
for hadronic showers, the correlated fluctuations between the longitudinal and lateral energy distributions.
The energy that is deposited in the calorimeter depends on the kinetic energy ( _𝐸_ kin ) of the particle and is
the energy used for parameterization unless otherwise specified. The _𝐸_ kin is defined as the particle energy
minus its mass. For antiprotons and antineutrons, the rest mass is added instead of being subtracted as their
annihilation will result in additional energy deposited in the calorimeter.


The simulation of the total shower energy and its longitudinal distribution between layers, including
correlations, provides an approximate simulation of jets, electrons, photons and _𝜏_ -leptons, albeit with
overestimated reconstruction and identification efficiencies. The simulation of the average lateral energy
spread plays an important role in the reconstruction and identification of objects for physics analysis. For
speed, a simplified geometry of the calorimeter cells (see Section 2) is used in AtlFast3, where each cell


3


belongs to a longitudinal sampling layer of the calorimeter and is either a cuboid in _𝜂_, [1] _𝜙_, and _𝑟_ for the
layers in the detector barrel, or a cuboid in _𝜂_, _𝜙_, and _𝑧_ for the layers in the detector endcaps up to | _𝜂_ | _<_ 3 _._ 2,
or a cuboid in _𝑥_, _𝑦_, and _𝑧_ for the forward calorimeter layers. This means that the accordion structure of the
real ATLAS liquid-argon electromagnetic calorimeter must be emulated. The improved average shower
energy distribution and a correction for the accordion structure in AtlFast3 reproduces the reconstruction
and identification efficiencies of the Geant4 simulation, especially for electrons and photons.


Fluctuations also play an important role in the calorimeter simulation. The simulation of independent and
uncorrelated energy fluctuations in calorimeter cells in individual showers relative to average showers
is required for an accurate description of electrons and photons of all energies. Correct modelling of
the fluctuations is also crucial for hadrons, where at low energy the fluctuations in the lateral energy
distribution are dominated by sampling fluctuations, noise and additional proton–proton interactions. For
medium-energy hadrons, correlated fluctuations between the longitudinal and lateral energy distributions in
hadronic showers play an important role and are accurately simulated by FastCaloGAN (see Section 5).


AtlFast3 combines the strengths of the FastCaloSim V2 and FastCaloGAN calorimeter simulation approaches.
The updated calorimeter parameterization in FastCaloSim V2 is used to simulate electromagnetic showers
and hadronic showers with low and high energies while Generative Adversarial Networks (GANs) [8]
trained for FastCaloGAN [9] are introduced for medium-energy hadrons because of their ability to model
correlated fluctuations. The two tools are combined to optimize the performance of the reconstruction
and a smooth interpolation between them is performed. Both calorimeter simulation approaches are
derived using single particles simulated with Geant4 to model the shower development in the ATLAS
electromagnetic and hadronic calorimeters. As the calorimeters do not necessarily contain the full showers,
the rate at which secondary particles punch through to the muon spectrometer is parameterized and those
particles are simulated with Geant4. The parameterization of muon punch-through is a new feature in
AtlFast3. Geant4 is also used to simulate particles in the inner detector and hadrons with energies below a
few hundred MeV in the calorimeters.


Section 2 introduces the ATLAS detector and provides details about the detectors used in AtlFast3. Datasets
and samples are presented in Section 3. Sections 4 and 5 discuss the FastCaloSim V2 and FastCaloGAN
parameterizations, respectively. The simulation of muon punch-through is discussed in Section 6. Section 7
discusses how AtlFast3 is constructed by combining FastCaloSim V2 and FastCaloGAN. The performance
of AtlFast3 in the reconstruction and in physics analysis is discussed in Section 8. Section 9 concludes the

paper.

#### **2 The ATLAS detector**


The ATLAS detector [1, 10] at the LHC covers nearly the entire solid angle around the collision point. It
consists of an inner tracking detector surrounded by a thin superconducting solenoid, electromagnetic and
hadron calorimeters, and a muon spectrometer incorporating three large superconducting air-core toroidal
magnets. An extensive software suite [11] is used in the reconstruction and analysis of real and simulated
data, in detector operations, and in the trigger and data acquisition systems of the experiment.


1 ATLAS uses a right-handed Cartesian coordinate system with its origin at the nominal interaction point (IP) in the centre of
the detector. The _𝑧_ -axis is along the beam pipe, and the _𝑥_ -axis points from the IP to the centre of the LHC ring. Cylindrical
coordinates ( _𝑟_, _𝜙_ ) are used in the transverse plane, _𝜙_ being the azimuthal angle around the beam pipe. The rapidity is defined as
_𝑦_ = (1/2) ln[( _𝐸_ + _𝑝_ _𝑧_ )/( _𝐸_ − _𝑝_ _𝑧_ )], while the pseudorapidity is defined in terms of the polar angle _𝜃_ as _𝜂_ = − ln tan( _𝜃_ /2).


4


**2.1 Inner detector**


The inner detector is immersed in a 2 T axial magnetic field and provides charged-particle tracking in the
range | _𝜂_ | _<_ 2 _._ 5. The high-granularity silicon pixel detector is closest to the collision point and typically
provides four measurements per track, the first hit normally being in the innermost layer. It is followed
by the silicon microstrip tracker, which usually provides eight measurements per track. These silicon
detectors are complemented by the transition radiation tracker (TRT), which enables radially extended track
reconstruction up to | _𝜂_ | = 2 _._ 0. The TRT also provides electron identification information based on the
fraction of hits (typically 30 in total) above a higher energy-deposit threshold corresponding to transition
radiation.


**2.2 Calorimeters**


The calorimeter system covers the pseudorapidity range | _𝜂_ | _<_ 4 _._ 9 and exploits several technologies to
measure the energy deposited by different types of particles. In Table 1 the different calorimeter modules,
different layers in the radial direction, the acronyms used to refer to each layer, the coverage in _𝜂_ by each
layer and the transition regions with gaps between different layers are summarized. In the transition regions
of the calorimeter, the detector response deteriorates relative to the rest of the acceptance.


Electromagnetic showers are measured by high-granularity lead/liquid-argon (LAr) calorimeters. The
electromagnetic barrel (EMB) and electromagnetic endcap calorimeters (EMEC) provide coverage within
the region | _𝜂_ | _<_ 3 _._ 2. There are three EMEC sampling layers in the precision-measurement region
(1 _._ 5 _<_ | _𝜂_ | _<_ 2 _._ 5) and two layers in the higher- | _𝜂_ | region (2 _._ 5 _<_ | _𝜂_ | _<_ 3 _._ 2). An additional thin LAr
Presampler covering | _𝜂_ | _<_ 1 _._ 8 corrects for energy loss in material upstream of the calorimeters. The
electromagnetic calorimeters have an accordion-shape geometry, which provides several active layers in a
compact design without any gaps. An illustration of this structure and of the segmentation in each layer is
shown in Figure 1. Each layer consists of a number of cells, and groups of cells are referred to as towers,
which are used by the trigger. The thickness of the calorimeter is given in units of radiation length, _𝑋_ 0 .


Hadronic showers are measured in a steel/scintillator-tile calorimeter segmented into three barrel structures
within | _𝜂_ | _<_ 1 _._ 7 (one TileBar and two TileExt), and two copper/LAr hadron endcap calorimeters (HEC) for


Table 1: The different calorimeter modules, their number of layers in the radial direction, the acronyms used to refer
to each layer and their coverage in _𝜂_ . The first layers of the electromagnetic calorimeter are Presampler layers and
have finer granularity than the subsequent layers.


Calorimeter Layers Module Name _𝜂_ -coverage Sampling Layer

4 Electromagnetic Barrel (EMB) | _𝜂_ | _<_ 1 _._ 5 PreSamplerB, EMB1, EMB2, EMB3
Electromagnetic calorimeters
4 Electromagnetic Endcap (EMEC) 1 _._ 5 _<_ | _𝜂_ | _<_ 1 _._ 8 PreSamplerE
1 _._ 5 _<_ | _𝜂_ | _<_ 3 _._ 2 EME1, EME2
1 _._ 5 _<_ | _𝜂_ | _<_ 2 _._ 5 EME3

4 Hadronic Endcap (HEC) 1 _._ 5 _<_ | _𝜂_ | _<_ 3 _._ 2 HEC0, HEC1, HEC2, HEC3
3 Tile Barrel (TileBar) | _𝜂_ | _<_ 1 _._ 0 TileBar0, TileBar1, TileBar2

Hadronic calorimeters

3 Tile Extended Barrel (TileExt) 0 _._ 8 _<_ | _𝜂_ | _<_ 1 _._ 7 TileExt0, TileExt1, TileExt2
3 Tile Gap (TileGap) 1 _._ 0 _<_ | _𝜂_ | _<_ 1 _._ 6 TileGap1, TileGap2, TileGap3
Forward calorimeter 3 FCal 3 _._ 1 _<_ | _𝜂_ | _<_ 4 _._ 9 FCal0, FCal1, FCal2

                    - between barrel and endcap | _𝜂_ | ≈ 1 _._ 45                    Transition regions - between outer and inner wheel of endcap | _𝜂_ | = 2 _._ 5                     - between endcap and FCal | _𝜂_ | ≈ 3 _._ 2                    

5


1 _._ 5 _<_ | _𝜂_ | _<_ 3 _._ 2. The granularity of HEC3 decreases for | _𝜂_ | _>_ 2 _._ 5. The transition region between the barrel
and the endcap has additional detectors, the Tile Gap layers.


Coverage at higher | _𝜂_ | (3 _._ 1 _<_ | _𝜂_ | _<_ 4 _._ 9) is provided by the forward calorimeter (FCal). Two different
technologies, copper/LAr and tungsten/LAr, are used in the FCal, which provides both for electromagnetic
and hadronic energy measurements.


The energy deposited in the calorimeters is read out from cells which define the granularity of the detector.
The granularity varies significantly depending on the layer and can vary with _𝜂_ and _𝜙_ .


Towers�in�Sampling�3�
Δ�× Δη�=�0.0245× 0.05







Figure 1: Granularity of the electromagnetic barrel LAr calorimeters [12]. The accordion geometry is indicated with
blue and orange lines on the side of the tower.


**2.3 Muon spectrometer**


The muon spectrometer (MS) comprises separate trigger and high-precision tracking chambers and
measures the deflection of muons in a magnetic field generated by the superconducting air-core toroidal
magnets. The field integral of the toroids ranges between 2.0 and 6.0 Tm across most of the detector. A
set of precision chambers covers the region | _𝜂_ | _<_ 2 _._ 7 with three layers of monitored drift tubes (MDTs),
complemented by cathode-strip chambers (CSCs) in the forward region, where the background is highest. In
the barrel region ( | _𝜂_ | ≤ 1 _._ 05), the MDT chambers are located in and around eight coils of superconducting
toroid magnets. In the endcap (1 _._ 05 ≤| _𝜂_ | ≤ 2 _._ 7) sections of the MS, the MDTs are located both in front of
and behind the endcap toroid magnets. The innermost detector in the endcap region is instrumented with
CSCs instead of MDTs to withstand higher rate and background conditions.


6


Secondary particles, created in showers in the calorimeters, leaking into the MS can have a significant
impact on muon reconstruction and depends on the calorimeter simulation. This effect is called muon
punch-through and the technique used to simulate it is described in Section 6.

#### **3 Dataset simulation and preprocessing**


This section discusses the simulation of the datasets used to derive the calorimeter parameterizations and
the datasets used to validate the performance of AtlFast3.


**3.1 Simulation of reference samples with Geant4**


The reference samples used to generate the calorimeter parameterizations discussed in this paper were
produced using Geant4 version 10.1.3 [5], which was released in 2016 and use the FTFP ~~B~~ ERT ~~A~~ TL
physics list [13]. FTFP ~~B~~ ERT ~~A~~ TL uses the Bertini intra-nuclear cascade model below 9 GeV and
transitions to the Fritiof model [14–16] with a pre-compound model for 12 GeV and higher. The default
Geant4 electromagnetic physics list is used. The standard configuration for ATLAS simulation in physics
analyses, referred to as the MC16 campaign, uses a fast simulation technique known as Frozen Showers [6]
to simulate electromagnetic showers in the FCal. This latter configuration has been used for full simulation
samples for all papers on Run 2 data published by ATLAS to date. The reference samples do not use the
Frozen Shower technique, but it is used for the validation samples discussed in Section 3.3.


The reference datasets consist of single-particle events produced on a cylinder with _𝑟_ = 1148 mm and
| _𝑧_ | _<_ 3550 mm located just outside the TRT because only particles reaching the calorimeters have to
be parameterized. This means that the impact of the cryostat and solenoid material is included in the
reference datasets. The particles are produced with directions of propagation consistent with production
at the interaction point and simulated without the spread of the LHC beam bunches, for positive and
negative values of _𝜂_ and with a uniform distribution in _𝜙_ . The impact of the spread of the beam bunches is
negligible because particles are parameterized according to where they enter the calorimeter. Photons
( _𝛾_ ) and electrons ( _𝑒_ [±] ) are used to parameterize electromagnetic showers, and positively and negatively
charged pions ( _𝜋_ [±] ) are used to parameterize hadronic showers. The positive and negative charge states of
electrons and pions are combined since the difference in shower development due to different charge is
negligible. Charged pions are used to model the simulation of hadrons because the dependence of the
hadronic showers on particle type is very small. The Geant4 simulation is run in a special configuration
with simulation steps which are smaller than those in the default configuration in the calorimeter so that
details of the spatial position of the deposited energy or hits are saved. These detailed hits are used to
parameterize lateral shower shapes in FastCaloSim V2, discussed in Section 4.2, and for the training of
FastCaloGAN. The sum of the energies of these detailed hits is lower than that of the standard Geant4 hits
due to a small bug in Geant4. This introduces a small bias in the energy distributions for FastCaloGAN
and also for FastCaloSim V2 standard hits in the derivation of the longitudinal energy parameterization
of FastCaloSim V2. The bias is non-negligible for photons and electrons, and this is corrected for in
FastCaloSim V2. After the correction, no significant impact on the physics performance of AtlFast3 is
observed. Deposited energies in simulation are digitized without considering electronic noise and cross-talk
between calorimeter cells in the readout electronics.


7


The calorimeter parameterization is obtained for 100 uniform _𝜂_ slices to provide coverage up to | _𝜂_ | = 5.
This range slightly exceeds the calorimeter _𝜂_ range to include particles depositing only a fraction of their
energy in the calorimeters. In each slice, up to 19 single-particle samples were produced, starting from
a minimum momentum of 16 MeV and up to 4.2 TeV with the momentum doubling for each sample.
The FastCaloGAN and FastCaloSim V2 parameterizations for pions are derived from the samples with
energies of 256 MeV and higher as discussed in Section 7.2.2. For photons, the samples with energies
of 64 MeV and higher are used for FastCaloSim V2 and those with energies of 128 MeV and higher
are used for FastCaloGAN. All samples are used to derive the energy interpolation splines discussed in
Section 7.3. Ten thousand events were produced for each of the samples with energies up to 256 GeV .
With increasing energies above 256 GeV, the number of events was progressively reduced, reaching 1000
events for the highest-energy sample, due to the significant increase in the time required to simulate the
events in Geant4.


Single-(anti)proton, single-(anti)neutron and single-(anti)kaon samples were produced in the same _𝜂_ and
momentum slices to derive the corrections described in Section 7.4.3. These corrections are sufficient to

provide good performance for these stable hadrons, thus avoiding a dedicated parameterization and thereby
reducing the memory footprint of AtlFast3.


**3.2 Voxelization**


The spatial energy deposits in each layer in the Geant4 datasets are grouped into volumes called ‘voxels’
for parameterization of FastCaloSim V2 and FastCaloGAN. Only layers with a significant amount of
energy, referred to as ‘relevant layers’, are considered in the parameterization. Relevant layers are defined
using criteria on the fraction of energy deposited in the layers with respect to the total energy deposited
in the calorimeters. The criteria used by the two simulators to determine the relevant layers are slightly
different:


   - In FastCaloSim V2, only layers with energy fractions larger than 0.1% are used; this procedure is
performed for each sample independently.


   - In FastCaloGAN all samples in the same _𝜂_ slice are processed with the same number of voxels. The
relevant layers are determined using only the 1 TeV energy point and have an energy fraction larger
than 0.1%. In addition, a layer with less energy is considered relevant if it is in front of a relevant
layer. For example, PreSamplerB is always included in FastCaloGAN in the barrel region of the
detector even if the energy deposited there is below the threshold.


The coordinates of hits in relevant layers, _𝜂_ [hit] and _𝜙_ [hit], are calculated relative to the extrapolated position of
the particle in that layer, _𝜂_ [extr] and _𝜙_ [extr] . The extrapolation is calculated from the momentum of the particle
at the point where it enters the calorimeter and propagated through the calorimeter, taking into account
the magnetic field for charged particles. Equation (1) shows the relative angular coordinates, Δ _𝜙_ and Δ _𝜂_,
of the showers. The coordinates of the calorimeter cell associated with the hit, _𝑧_ cell and _𝑟_ cell, are used to
transform the relative hit coordinates into millimeters, Δ _𝜙_ [mm] and Δ _𝜂_ [mm], which provides a more convenient
description of the shower.


8


Table 2: The binning used for the voxelization of pions in the different calorimeter layers for FastCaloGAN in the
0 _<_ | _𝜂_ | _<_ 0 _._ 8 range. Each ellipsis indicates when the same binning continues until the subsequent listed number.


Layer Bin boundaries in Δ _𝑅_ ~~[mm]~~ [mm] Number of bins in _𝜙_
PreSamplerB 5, 10, 30, 50, 100, 200, 400, 600 1
EMB1 1, 4, 7, 10, 15, 30, 50, 90, 150, 200 10
EMB2 5, 10, 20, 30, 50, 80, 130, 200, 300, 400 10
EMB3 50, 100, 200, 400, 600 1
TileBar0 10, 20, 30, · · · 100, 130, 160, 200, 250 · · · 400, 1000, 2000 10
TileBar1 10, 20, 30, · · · 100, 130, 160, 200, 250, · · · 400, 600, 1000, 2000 10
TileBar2 0, 50, 100, · · · 300, 400, 600, 1000, 2000 1


Δ _𝜂_ = _𝜂_ [hit] − _𝜂_ [extr] _,_ (1)

Δ _𝜙_ = _𝜙_ [hit] − _𝜙_ [extr] _,_

Δ _𝜂_ [mm] = Δ _𝜂_ × _𝜂_ Jacobi × ~~√~~ _𝑟_ cell [2] [+] _[ 𝑧]_ [2] cell _[,]_

Δ _𝜙_ [mm] = Δ _𝜙_ × _𝑟_ cell _,_


where _𝜂_ Jacobi = |2 × exp(− _𝜂_ extr. )/(1 + exp(−2 _𝜂_ extr. )) |.



Δ _𝑅_ [mm] =
~~√~~



(Δ _𝜂_ [mm] ) [2] + (Δ _𝜙_ [mm] ) [2] _,_ (2)



_𝛼_ = arctan2 (Δ _𝜙_ [mm] _,_ Δ _𝜂_ [mm] ) _._


The hit positions are then transformed to polar coordinates defined in Eq. (2) and grouped into voxels of
different size:


   - In FastCaloSim V2, the shower symmetry along _𝜙_ with respect to the centre of the shower in each
layer is exploited. A binning of 1 mm in the radial direction is used in the high-granularity EMB1
and EME1 layers while 5 mm is used in the other layers. Along the angular direction, _𝛼_, eight
uniform bins are used in all relevant layers. These settings are used for all particles. The size of each
voxel is much smaller than the calorimeter cell dimensions.


   - In FastCaloGAN, the size of the voxel is optimized for each particle type and detector _𝜂_ slice. In the
radial direction, a variable bin width is used with increasingly wider bins. An example is shown in
Table 2 for pions in the barrel. Only layers with a large fraction of the total energy are binned along
the angular direction. The angular positions of the showers in the other layers are neglected and
simulated uniformly. Ten bins of equal size are used for layers binned in the angular direction. Due
to the variable-width bins in the radial direction, the size of the voxels can be significantly larger than
the cell dimensions. This voxel definition is optimized for an accurate training of the GANs since
using as many bins as FastCaloSim V2 would significantly increase the training time and instability
of the GANs, ultimately reducing the performance of FastCaloGAN.


9


**3.3 Validation datasets**


A range of different Monte Carlo samples commonly used for physics performance studies and physics
analysis are used to validate the performance of AtlFast3. Table 3 summarizes the key samples, which are
also discussed in this section. The matrix element (ME) order describes the precision at which the process
is produced by the generator in perturbative quantum chromodynamics (QCD); this can be leading order
(LO), next-to-leading order (NLO) or next-to-next-to-leading order (NNLO).


The production of _𝑡𝑡_ ¯ events was performed with Powheg Box r2330 [17] interfaced with Pythia 6.427 [18]
for the parton shower and hadronization modelling with the CT10 [19] set of parton distribution functions
(PDFs) and the Perugia2012 set of tuned parameters (P2012 tune) [20]. At least one of the top quarks is
required to produce a lepton when decaying. This sample can be used to validate small-radius jets, leptons
and _𝑏_ -jets.


Events containing _𝑍_ bosons decaying into a pair of electrons, muons or _𝜏_ -leptons were generated with
Powheg Box r2856 [21] at NLO in QCD using the CTEQ6L1 [22] PDF set. The events were interfaced
with Pythia 8.186 [23] for the parton shower and hadronization modelling using the AZNLO tune [24].
The samples were generated with _𝑝_ T ( _𝑍_ ) = 0. These samples are used to validate electrons, muons and
_𝜏_ -leptons.


Events containing a new hypothetical spin-1 boson, _𝑊_ [′], decaying into a _𝑊𝑍_ pair, which subsequently
decay into hadrons, were generated using Pythia 8.235 with the NNPDF23LO [25] PDF set. The _𝑊_ [′]

bosons were generated with a mass of 13 TeV, and the differential cross-section is reweighted to have a flat
distribution of jet _𝑝_ T from 200 GeV to 3 TeV . Similarly, a sample of _𝑍_ [′] bosons with a mass of 4 TeV was
generated using Pythia 8.235, and the _𝑍_ [′] boson was subsequently decayed into a top and anti-top quark
pair. The top quarks were forced to decay into hadrons in the samples, which allows the substructure of
jets to be validated. Similar to the _𝑊_ [′] sample, the differential cross-section is reweighted to have a flat
distribution of jet _𝑝_ T from 200 GeV to 3 TeV to better populate kinematic regions with higher jet _𝑝_ T . These
samples are used to validate the substructures of various jets with very high transverse momentum.


Samples of multijet events were simulated with the Pythia 8.186 general-purpose event generator interfaced
to EvtGen 1.2.0 [26] for decay of heavy-flavour mesons. The NNPDF23 PDF set [27] and the A14
tune [28] were used. The rapidly falling spectrum of leading-jet momenta requires this simulation to be
filtered in leading-jet _𝑝_ T . These samples are used to validate jets in a range of _𝑝_ T regimes.


Higgs boson production via gluon–gluon fusion (ggF) was simulated at NNLO accuracy in the strong
coupling constant _𝛼_ s using Powheg NNLOPS [29–33], which achieves NNLO accuracy for arbitrary
inclusive _𝑔𝑔_ → _𝐻_ observables by reweighting the Higgs boson rapidity spectrum of MJ-MiNLO [34–36]
to that of HNNLO [37]. The PDF4LHC15 PDF set [38] and the AZNLO tune of Pythia 8 were used. This
simulation was interfaced with Pythia 8.230 for parton shower and non-perturbative hadronization effects.
The Higgs boson was decayed into a pair of photons.


The impact of pileup in the same and neighbouring proton-bunch crossings was modelled by combining
detector signals from simulated inelastic _𝑝𝑝_ events with the hard-scattering (HS) event [39]. These pileup
events were generated with Pythia 8.186 [23] using the NNPDF23 set of PDFs and the A3 tune [40]. The
pileup events were simulated using Geant4 even for samples produced with AtlFastII or AtlFast3, and the
same pileup events were reused for all samples. An average number of pileup interactions per _𝑝𝑝_ bunch
crossing of 38 with a standard deviation of 12 was used, similar to the pileup distribution recorded by the
ATLAS experiment during Run 2.


10


The validation samples are reconstructed using the standard algorithms for the ATLAS experiment [41–46].
The energy scale and resolution of reconstructed leptons and jets, as well as their reconstruction and
identification efficiencies in the simulation are corrected to match those measured in data using the standard
procedures of the ATLAS experiment [47]. Unless stated otherwise, the same reconstruction code and
corrections are applied to samples simulated with Geant4, AtlFastII and AtlFast3.


Table 3: Summary of the Monte Carlo generator settings for the simulation of samples for validation of AtlFast3. See
text for details.


Process Generator ME Order PDF Parton Shower Tune


SM process samples


_𝑡𝑡_ ¯ Powheg Box r2330.3 NLO CT10 Pythia 6.427 P2012


_𝑍_ _[★]_ / _𝛾_ _[★]_ → _𝜏𝜏_ Powheg Box r2856 NLO CTEQ6L1 Pythia 8.186 AZNLO


_𝑍_ → _𝜇_ [+] _𝜇_ [−] Powheg Box r2856 NLO CTEQ6L1 Pythia 8.186 AZNLO


_𝑍_ → _𝑒𝑒_ Powheg Box r2856 NLO CTEQ6L1 Pythia 8.186 AZNLO


_𝑊_ [′] (13 TeV) → Pythia 8.235 LO NNPDF23LO Pythia 8.235 A14
_𝑊𝑍_ → 4 _𝑞_


_𝑍_ [′] (4 TeV) → _𝑡𝑡_ ¯ Pythia 8.235 LO NNPDF23LO Pythia 8.235 A14


Dijet: leading jet Pythia 8.186 NLO NNPDF23 Pythia 8.186 A14
_𝑝_ T = 140–400 GeV


Dijet: leading jet Pythia 8.186 NLO NNPDF23 Pythia 8.186 A14
_𝑝_ T = 1.8–2.5 TeV


ggF Higgs → _𝛾𝛾_ Powheg v2 NNLOPS NNLO PDF4LHC15 Pythia 8.230 AZNLO

#### **4 Calorimeter simulation with FastCaloSim V2**


FastCaloSim V2 parameterizes the longitudinal and lateral development of showers in the calorimeter.
During AtlFast3 simulation, energy is directly deposited in calorimeter cells using the parameterized
responses. The longitudinal parameterization along with a correction to the energy resolution is discussed
in Section 4.1. Parameterization of the average lateral shower distribution is discussed in Section 4.2.
Finally, the simulation of hits using longitudinal and average lateral shower parameterization is described
in Section 4.3.


**4.1 Longitudinal shower development**


As particles shower in the calorimeter, they deposit energy in the various layers. The amount of energy
deposited in each layer depends on how deep in the calorimeter the shower was initiated. The amount of
energy deposited is highly correlated between layers, making it difficult to independently parameterize the
response for each layer.


Principal Component Analysis (PCA) [48] is used to classify showers from the samples introduced in
Section 3 for each slice of energy, _𝜂_ bin, and particle type. The PCA transformation is performed twice.
The initial PCA, referred to as the ‘first PCA’, is used to classify showers into bins referred to as ‘PCA


11


bins’. A second PCA transformation, referred to as the ‘second PCA’, is performed in each bin of the first
PCA to generate uncorrelated and approximately Gaussian distributions. These Gaussian distributions
from each PCA bin are used in the FastCaloSim V2 simulation. The steps of this PCA chain are discussed
in detail below.



















(a)



(b)











(c)


Figure 2: Example of the steps in the first PCA transformation for 65 GeV photons with 0 _._ 2 _<_ | _𝜂_ | _<_ 0 _._ 25 : (a) shows
the distribution of energy fractions in EMB1, (b) the Gaussian distribution, and (c) is the leading principal component
of the first PCA with bin borders (dotted pink lines) showing five PCA bins. The steps of the second PCA are identical
to those of the first PCA but performed in each PCA bin separately to generate uncorrelated Gaussian distributions
using all principal components of the second PCA. The errors bars indicate the size of the statistical uncertainty.


The distribution of the fraction of energy deposited (see Figure 2(a)) in each calorimeter layer and the total
energy deposited (summed over all layers) are used to classify the showers. Only the relevant layers as
defined in Section 3.2 are considered. The energy fraction in each layer is integrated and transformed into
a Gaussian distribution using a cumulative distribution function transformation (see Figure 2(b)). These
Gaussian distributions from each layer and each event are used to construct a PCA matrix to perform the
first PCA.


The first PCA converts the set of correlated energies into a set of linearly uncorrelated quantities by an
orthogonal transformation of the coordinate system. The transformation is calculated using the covariance
and the eigenvectors of the PCA matrix. The principal components with highest and second-highest


12


variance are referred to as the leading and sub-leading principal components of the first PCA. Figure 2(c)
shows the leading principal component of the first PCA. To classify the shower, the leading and in some
cases the sub-leading principal component of the first PCA is binned in equally populated PCA bins
(covering equal ranges of cumulative PCA bin probability). A bin with zero deposited energy is included,
as this improves the modelling of low-energy particles. Typically, five PCA bins in the leading principal
component of the first PCA are used (see Figure 2(c).) However, within the transition regions of the
calorimeter layers, given in Table 1, the sub-leading principal component is also used in order to determine
PCA bins in two dimensions. The exact number of bins in each region is determined from a _𝜒_ [2] test giving
the best modelling of all energy fractions. The first PCA removes non-linear correlations between layers
and roughly classifies the showers according to their depth.


The effectiveness of the PCA transformation is demonstrated in Figures 3 and 4, which show the correlations
before and after the first PCA. The correlations between different layers are calculated from the Gaussian
inputs of the PCA matrix. After the first PCA, the correlations are calculated using a subset of the principal
components and are strongly reduced.


The total energy and the energy fractions in each first-PCA bin are transformed into Gaussian distributions
following the same method and then the second PCA is performed. The steps of the second PCA are
identical to those of the first PCA except that only the events in a given first-PCA bin are used. The
second-PCA rotation removes any further correlations in each first-PCA bin to produce uncorrelated
Gaussian outputs using all principal components. The mean and RMS of these Gaussian distributions, the
PCA matrices, the PCA bin probabilities and the inverse cumulative distributions are stored and used in the
simulation.


During simulation, the steps of the PCA chain are executed in reverse. For each simulated particle, a
PCA bin is selected using random numbers distributed according to the PCA bin probabilities. The
uncorrelated Gaussian distributions, stored in the parameterization, in the selected PCA bin are used to
generate uncorrelated random numbers. These random numbers are rotated using the inverse PCA matrix
of the second PCA to generate correlated random numbers. The correlated random numbers are then
mapped back to the total energy and the energy fractions deposited in each layer using the error function
and the inverse cumulative distributions of the first PCA.


The validation of the longitudinal energy parameterization is shown in Figure 5 for 65 GeV photons with
0 _._ 2 _<_ | _𝜂_ | _<_ 0 _._ 25. Incoming photons with an energy of 65 GeV are simulated using the FastCaloSim V2
parameterization and compared with Geant4 simulation. In this case, no digitization or reconstruction
algorithms are applied, but the energy deposited in the active regions of the calorimeter has been scaled by
the sampling fraction. This simulation without digitization and reconstruction is referred to as ‘stand-alone
simulation’.


In general, the energy fractions deposited in each calorimeter layer using the FastCaloSim V2 parameterization are observed to be in good agreement with Geant4. However, for the total energy distribution, a
residual difference in the mean and a larger RMS are observed. These small differences can impact the
modelling of complex quantities, e.g. the Higgs boson invariant mass distribution reconstructed from two
photons. Additional corrections are therefore applied to improve the modelling of both the resolution and
the mean, as discussed in Sections 7.4.1 and 7.4.3.


13


(a)



(b)

















(c)


Figure 3: Correlations between the transformed energies deposited in several layers, before PCA rotation, showing
(a) Presampler barrel vs EM barrel 1, (b) Presampler vs EM barrel 2 and (c) EM barrel 1 vs EM barrel 2. The
energies were transformed into Gaussian distributions. The correlation factors obtained from these 2D histograms
are displayed.



**4.2 Average lateral shower shape**


The lateral shower shape describes the lateral energy distribution in each calorimeter layer. The parameterization is derived in each relevant layer and for each PCA bin. The shower development is parameterized in
voxels using the coordinates defined in Eq. (2). To exclude hits far away from the centre of the shower, only
99.5% of the total energy of each PCA bin cumulatively along Δ _𝑅_ [mm] is considered. The shower shape
distribution in each PCA bin (of each layer) is then normalized to the energy in that PCA bin to create
the probability density function for the average shower shape. Figure 6 shows the average lateral shower
profile corresponding to the electromagnetic and hadronic showers in the second layer of the EM barrel
and the Tile barrel, respectively. The memory footprint of these histograms is reduced by storing only the
|Δ _𝜙_ [mm] | coordinates for 0 ≤ _𝛼_ ≤ _𝜋_, because the shower is symmetric in Δ _𝜙_ [mm] .



14


(a)



(b)



















(c)


Figure 4: Correlations between the first-PCA components after the PCA rotation. The individual components are
approximately Gaussian distributed.



**4.3 Simulation of hits**


A key limitation of AtlFastII is that the lateral shower shape simulation is based on the average shower
shape. This model works well for electrons and photons, but cannot reproduce the complex structure
of hadronic showers. ATLAS extracts the shower structure of electrons, photons, hadrons and jets by
clustering calorimeter cells using the TopoCluster algorithm [49]. The clustering proceeds by starting
from a seed cell with an energy 4 _𝜎_ above the calorimeter noise threshold and adding cells with an energy
at least 2 _𝜎_ above the noise and finally adding adjacent cells of any energy. However, using the average
shower shape means that the energy distribution and position of the hadronic clusters differ compared to
Geant4. Instead of directly using the average shower shape, FastCaloSim V2 uses the average shower
shape as a probability distribution function (pdf) to generate hits which are subsequently mapped onto the
calorimeter cell structure. For particles entering the calorimeter with a non-zero angle with respect to the
calorimeter cell boundaries, the position of each hit is modified to account for the longitudinal position
within each layer to improve the simulation of the shower shapes. For each PCA bin, the average value of
the longitudinal position distribution from the reference sample is used to correct the lateral position at
which the hits are produced. The models used to assign energy to each hit are discussed in the following


15


(a)



(b)













(d)



(c)



Figure 5: Validation of the energy, _𝐸_, parameterization is shown for 65 GeV photons with 0 _._ 2 _<_ | _𝜂_ | _<_ 0 _._ 25,
comparing the input Geant4 sample (black triangles) with FastCaloSim V2 (red dots). Good agreement is observed
for all layers and the total energy. The errors bars indicate the size of the statistical uncertainty.


sections. The energies of hits are normalized so that their sum exactly matches the simulated energy in a
layer as discussed in Section 4.1.



**4.3.1 Electrons and photons**


The number of generated hits for electrons and photons is calculated from the energy deposited in each
calorimeter layer and the intrinsic resolution of the calorimeter technology in that layer. For a given energy
_𝐸_ simulated within a calorimeter layer, the resolution is defined as:



_𝜎_ _𝐸_ / _𝐸_ = _𝑎_ /√︁



_𝐸_ /GeV ⊕ _𝑐_ (3)



where _𝑎_ is the stochastic term and _𝑐_ the constant term. The values used for _𝑎_ and _𝑐_ for the different detector

technologies are listed in Table 4 and are used to calculate the resolution, _𝜎_ _𝐸_ .


The expected number of hits, _𝜆_, which would produce this resolution from a Poisson statistical process is
calculated as:


16


(a) photon



(b) pion





Figure 6: The lateral shower shape parameterization for (a) photons and (b) pions with energies of 265 GeV in the
range 0 _._ 55 _<_ | _𝜂_ | _<_ 0 _._ 60 and parameterized in the second layer of the EM barrel and Tile barrel respectively. To
visualize the core of the shower, these plots have a cut-off at Δ _𝑅_ [mm] ∼ 100 mm.


Table 4: Stochastic and constant terms for the intrinsic calorimeter energy resolution for the different detectors and
used to simulate the hits for electrons and photons [12].


Calorimeter technology Stochastic term _𝑎_ Constant term _𝑐_


LAr EM barrel and endcap 10.1% 0.2%
Tile 56.4% 5.5%

LAr hadronic endcap 76.2% 0
FCal 28.5% 3.5%


_𝜆_ = 1/( _𝜎_ _𝐸_ / _𝐸_ ) [2] (4)


A random number following a Poisson distribution _𝑁_ = Poisson( _𝜆_ ) is used to simulate _𝑁_ hits of equal
energy _𝐸_ hit = _𝐸_ / _𝑁_ . The positions of these hits are randomly distributed according to the average shower
shape introduced in Section 4.2.


For electrons and photons, which deposit most of their energy in the LAr EM calorimeters, the expected
number of hits (see Eq. (4)) is dominated by the stochastic term, _𝑎_, in the energy resolution. Assuming
a stochastic term of 10.1% and equal energy for each hit, electron and photon showers have hits with
_𝐸_ hit ≈ 10 MeV.


**4.3.2 Hadrons**


The number of generated hits for hadrons is calculated following the same procedure as described in
Section 4.3.1. However, for hadrons, the stochastic and constant terms in each layer are _𝜂_ -dependent and


17


much larger due to intrinsic fluctuations in hadronic showers. To derive these terms, a special simulation of
charged pions is used, where in addition to the measurable energy deposited in the active material of the
calorimeter, the total energies lost in both the active and inactive parts of the calorimeter are recorded.
The ratio of these two energies in each layer is the sampling fraction per shower and varies with the total
energy deposited, denoted by _𝑓_ sample ( _𝐸_ ) . The relative resolution, _𝜎_ _𝐸_ / _𝐸_, of _𝑓_ sample ( _𝐸_ ) is fitted with Eq. (3)
to extract the stochastic and constant terms for each _𝜂_ -slice. Only showers that deposit more than 1 GeV of
energy in a calorimeter layer are considered in the fit.


The stochastic terms obtained from simulation for pions are in the range of approximately 30%–40% for the
EM calorimeters, 50%–60% for the Tile calorimeter, 60%–80% for the HEC calorimeter and 80%–100%
for the FCal. The constant terms _𝑐_ are in the range of 1%–10%.


Using _𝜂_ -dependent stochastic and constant terms significantly improves the modelling of hadronic showers
for most layers. A notable exception is observed in the highly granular calorimeter layers EMB1 and EME1.
In these cases, the stochastic and constant terms shown in Table 4 are used.


In Figure 7(a) the energy fractions inside voxels along Δ _𝑅_ [mm] in the EMB2 layer of the calorimeter are
shown for a 65 GeV charged pion in the range 0 _._ 20 _<_ | _𝜂_ | _<_ 0 _._ 25 for the first bin of the leading PCA using
Geant4. The number of voxels with a particular energy fraction is represented on the _𝑧_ -axis. In the
Geant4 distribution, away from the centre of the shower, only a small number of voxels have an energy of
_𝑂_ ( _𝐸_ hit ), while most voxels have a substantially lower energy. Due to the large stochastic terms for hadrons
( _>_ 30%), _𝐸_ hit is approximately 100–300 MeV for hadronic showers, which is similar to the energy needed
to seed a cluster.


In Figure 7(b) the same distribution is shown simulated using FastCaloSim V2 where each hit is assigned
equal energy. The mean and the RMS calculated by including the voxels with an energy of _𝑂_ ( _𝐸_ hit ), for
each distribution, are compared in Figure 7(d). Although the mean of the energy fraction is correctly
reproduced by FastCaloSim V2, the number of voxels with an energy of _𝑂_ ( _𝐸_ hit ) is substantially larger away
from the centre of the shower. In many cases just one of these hits together with some noise is sufficient to
seed the formation of a calorimeter cluster, which then leads to substantial differences in the cluster energy
and position distribution compared to Geant4.


To correct for this mismodelling, a second model is developed where instead of assigning equal energy, a
hit weight is introduced. The weight is calculated such that the number of hits simulated in a certain Δ _𝑅_ [mm]

bin is changed to better reproduce the RMS of the distribution from Geant4, denoted by RMS G4 . The
steps involved to calculate this weight are discussed below.


As a first step, the voxels with sufficiently low energy are assigned _𝐸_ voxel = 0. Only voxels that do not
change the mean by more than 0.01% are considered in this step. Then two Poisson distributions are
calculated, one reproducing the fraction of voxels with _𝐸_ voxel = 0 and a second one reproducing the RMS
of the total Geant4 distribution. The smaller of the two RMS values is used and denoted by RMS Poisson .


The RMS Poisson value is used to determine the number of simulated hits by calculating:


_𝑁_ Poisson = 1/(RMS Poisson / _𝜆_ ) [2] _._


The weight is then calculated using _𝑁_ Poisson as follows:


_𝑤_ = ⟨ _𝐸_ voxel / _𝐸_ hit ⟩/ _𝑁_ Poisson


and the energy is recalculated as:
_𝐸_ [′]
hit [=] _[ 𝐸]_ [hit] [ ·] _[ 𝑤.]_


18


(a) Geant4



(b) Model: equal hit energy



















(c) Model: weighted hit energy



(d) Comparison of mean and RMS of the models



Figure 7: Ratio _𝐸_ voxel / _𝐸_ hit as function of Δ _𝑅_ [mm] for deposited energy from a 65 GeV charged pion in EMB2 in the
range 0 _._ 20 _<_ | _𝜂_ | _<_ 0 _._ 25 in the first bin of the leading PCA (PCA=1). Entries with _𝐸_ voxel = 0 are shown in the
underflow bin below 10 [−][9] . Lateral shower shape model (a) in Geant4, (b) in a model using equal deposited energies,
(c) in a model using weighted hit deposited energies. (d) Comparison of the mean (central value) and the RMS (error
bars) for the equal hit, weighted hit and Geant4 models. The yellow band indicates the 1 _𝜎_ uncertainty for Geant4.


These weights are calculated for each average shower shape discussed in Section 4.2. To ensure that the
average shower shape is unchanged, a correction of 1 / _𝑤_ is applied to the probability of all voxels at a
distance of Δ _𝑅_ [mm] from the shower center.


In addition, if the RMS of the Poisson distribution is smaller than that of Geant4, i.e. RMS Poisson _<_ RMS G4,
additional fluctuations are added by applying a smearing function to the _𝑁_ Poisson value. The smearing
function has the form e _[𝑠]_, where _𝑠_ is a random number generated from a Gaussian distribution such that:



RMS e _[𝑠]_ = RMS smearing / ~~√~~ _𝑁_ Poisson _,_



and RMS smearing is calculated as:



RMS [2]
smearing [=][ RMS] [2] G4 [−] [RMS] [2] Poisson _[.]_



Combining these corrections, the hit energy is derived as:


_𝐸_ [′′]
hit [=] _[ 𝐸]_ [hit] [ ·] _[ 𝑤]_ [·][ e] _[𝑠]_ _[.]_



19


Figure 7(c) shows the _𝐸_ voxel / _𝐸_ hit distribution simulated using FastCaloSim V2 with the weighted hit model.
The number of voxels with an energy of _𝑂_ ( _𝐸_ hit ) is seen to be substantially better modelled when compared
with Geant4. Additionally, the mean and the RMS of the equal hit model and the weighted hit model
are overlaid with those of Geant4 in Figure 7(d), demonstrating the improved modelling of the RMS for
weighted hits. The dependence of the weight _𝑤_ and the RMS e _[𝑠]_ parameter is stored as function of Δ _𝑅_ [mm]

together with the average shower shape scaled by 1 / _𝑤_ for all charged pions with energy above 1 GeV . For
hadrons, the weighted hit model is used instead of the equal hit energy model.

#### **5 Calorimeter simulation with FastCaloGAN**


FastCaloGAN is a fast calorimeter simulation tool that parameterizes the interactions of particles in the
ATLAS calorimeter system using 300 GANs, one for each particle type and _𝜂_ slice in which the reference
samples are produced. FastCaloGAN takes a different approach than FastCaloSim V2, which as seen in
Section 4 factorizes the shower parameterization into several components, i.e. longitudinal and lateral
energy distributions for different energy points that requires interpolation between them. A GAN, instead,
provides a comprehensive solution to the simulation of any particle of any energy. This results in a
simpler model that has a lower memory requirement at the price of a significantly higher time needed for
producing the parametrisation used in the simulation. A detailed description of FastCaloGAN is provided
in Ref. [50], and other studies of deep generative models for fast calorimeter simulation can be found in
Refs. [51–56]. GANs were chosen because they have proven successful in generating realistic showers in
calorimeters. A GAN [8] is a combination of two deep networks, a generator producing artificial showers
and a discriminator trying to distinguish the generated images from real ones. The two networks compete
against each other in a game resulting in a type of training that is unlike those for other machine-learning
problems. For example, the loss functions cannot easily be used to assess the quality of the training or
to select the best training point. The architecture of the GANs is described in Section 5.1. Section 5.2
discusses the training strategy, the selection of the best epoch and its performance. Finally, the strategy to
map the energy from the voxels to the calorimeter cells is explained in Section 5.3.


**5.1 Architecture**


FastCaloGAN uses the Wasserstein GAN [57] with a gradient penalty (WGAN-GP) term [58] in the loss
function of the discriminator. This configuration provides good performance and training stability.


The WGAN-GP is implemented in TensorFlow 2.0 [59] such that the training can be performed on either
CPUs or GPUs. The architecture of the WGAN-GP is presented in Figure 8. The generator uses a latent
space of 50 values and has three hidden layers of increasing size. The output layer of the generator and
the input layer of the discriminator have a number of nodes equal to the number of voxels (NVoxel)
corresponding to the specific particle type and _𝜂_ slice. The discriminator maintains the same number of
nodes until the last layer, which has a single output node. The GANs are conditioned on a single parameter,
the true momentum of the particle.


Each node uses the Rectified Linear Unit (ReLU) activation function. Both the generator and discriminator
use the Adam [60] optimizer with a learning rate of 10 [−][4] . The exponential decay rate for the first moment
( _𝛽_ 1) is set to 0.5, while the second moment ( _𝛽_ 2) is set to the default value (0.999), as are all other
parameters that are explicitly given here. The training is performed using a batch size of 128 events, and


20


Conditional WGAN-GP
























|Generator<br>Latent Dense Dense Dense Dense Output<br>Space (50) 50 100 200 NVoxel<br>ReLU ReLU ReLU ReLU<br>True momentum Concatenate<br>Discriminator<br>Dense Dense Dense Dense<br>Output<br>NVoxel NVoxel NVoxel NVoxel<br>Data<br>Linear ReLU ReLU ReLU|Col2|Col3|Col4|
|---|---|---|---|
|Dense<br>50<br>ReLU<br>Dense<br>100<br>ReLU<br>Dense<br>200<br>ReLU<br>Dense<br>NVoxel<br>ReLU<br>Latent<br>Space (50)<br>Dense<br>NVoxel<br>Linear<br>Dense<br>NVoxel<br>ReLU<br>Dense<br>NVoxel<br>ReLU<br>Dense<br>NVoxel<br>ReLU<br>**Discriminator**<br>**Output**<br>**Generator**<br>**Output**<br>**Data**<br>True momentum<br>Concatenate|Dense<br>NVoxel<br>Linear<br>Dense<br>NVoxel<br>ReLU<br>Dense<br>NVoxel<br>ReLU<br>Dense<br>NVoxel<br>ReLU|Dense<br>NVoxel<br>Linear<br>Dense<br>NVoxel<br>ReLU<br>Dense<br>NVoxel<br>ReLU<br>Dense<br>NVoxel<br>ReLU|Dense<br>NVoxel<br>Linear<br>Dense<br>NVoxel<br>ReLU<br>Dense<br>NVoxel<br>ReLU<br>Dense<br>NVoxel<br>ReLU|
|Dense<br>50<br>ReLU<br>Dense<br>100<br>ReLU<br>Dense<br>200<br>ReLU<br>Dense<br>NVoxel<br>ReLU<br>Latent<br>Space (50)<br>Dense<br>NVoxel<br>Linear<br>Dense<br>NVoxel<br>ReLU<br>Dense<br>NVoxel<br>ReLU<br>Dense<br>NVoxel<br>ReLU<br>**Discriminator**<br>**Output**<br>**Generator**<br>**Output**<br>**Data**<br>True momentum<br>Concatenate|Dense<br>NVoxel<br>Linear<br>Dense<br>NVoxel<br>ReLU<br>Dense<br>NVoxel<br>ReLU<br>Dense<br>NVoxel<br>ReLU|||
|Dense<br>50<br>ReLU<br>Dense<br>100<br>ReLU<br>Dense<br>200<br>ReLU<br>Dense<br>NVoxel<br>ReLU<br>Latent<br>Space (50)<br>Dense<br>NVoxel<br>Linear<br>Dense<br>NVoxel<br>ReLU<br>Dense<br>NVoxel<br>ReLU<br>Dense<br>NVoxel<br>ReLU<br>**Discriminator**<br>**Output**<br>**Generator**<br>**Output**<br>**Data**<br>True momentum<br>Concatenate|Dense<br>NVoxel<br>Linear<br>Dense<br>NVoxel<br>ReLU<br>Dense<br>NVoxel<br>ReLU<br>Dense<br>NVoxel<br>ReLU|||



Figure 8: Schematic representation of the architecture of the GANs used by FastCaloGAN. The input to the generator
is at the top left and the output from the discriminator is at the bottom left. The Rectified Linear Unit (ReLU)
activation function is used in all layers of the discriminator with the exception of the last.


Table 5: Overview of the parameters of the WGAN-GP.


NVoxel Number of voxels

Generator nodes 50, 50, 100, 200, NVoxel

Discriminator nodes NVoxel, NVoxel, NVoxel, NVoxel, 1

Activation function ReLU

Optimizer Adam [60]
Learning rate 10 [−][4]

_𝛽_ 1 0.5
_𝛽_ 2 0.999

Batch size 128

Training ratio (D/G) 5
Gradient penalty ( _𝜆_ ) 10


the discriminator is trained five times for each training of the generator. Finally, the gradient penalty, _𝜆_, is
set to 10. These parameters are summarized in Table 5. This set of hyperparameters as well as the overall
architecture were chosen as a compromise between the modelling performance and the time required to
train the 300 GANs.


**5.2 Training**


Each GAN is trained first on a single energy point and then the other energy points are added progressively
to the training mixture starting from the energy points closest in energy to the initial sample. The training
procedure can be summarized as follows:


1. Train the first 50 000 epochs using the 32 GeV sample.


21


2. Every 20 000 epochs add a new sample, alternating between higher and lower energies.


3. Once all energy points have been added, continue training with all samples for the remaining epochs.


The energy in each voxel is normalized to the true momentum of the primary particle entering the
calorimeter, which means that the GAN only needs to learn the relative shape of the showers. The true
momenta, which are used as labels for the conditioning, are also normalized to the highest value (4.2 TeV ),
which results in a range of values (0,1] which is optimal for the training of the GANs.


The training is performed for 1 million epochs with a TensorFlow checkpoint saved every 1000 epochs to
monitor the improvements in the training. The training time for each GAN is approximately 8 hours on
the NVIDIA V100 [61] GPUs available on the CERN HTCondor system [62]. The limited number of
GPUs available to train the 300 GANs sets the limit of 1 million epochs, while the frequency at which the
checkpoints are stored is limited by both speed and disk space.


**5.2.1 Best epoch selection**


Due to the interplay between the generator and discriminator, the final epoch is not necessarily the best one.
The figure of merit used to select the best epoch is a _𝜒_ [2] between the reference samples and the GANs. The
variable chosen is the sum of the energy in all voxels that corresponds to the total energy deposited in the
calorimeter by the particle.


For each energy point, the range used for the distribution is defined to be a ± 3 RMS interval around the
peak for electromagnetic showers in the Geant4 reference samples. As the energy distributions of the
pions have longer tails, the range for hadrons is defined to be between − 4 RMS and 3.5 RMS. A total of 30
bins are used for all energy points. The _𝜒_ [2] is then evaluated between the binned distributions produced
from all events in the reference samples and 10 000 events generated from the GAN and weighted by the
statistical uncertainty. The overflow and underflow bins are not used in the _𝜒_ [2] evaluation. The total _𝜒_ [2] for
a checkpoint is the sum of the _𝜒_ [2] for each of the 15 energy points. The checkpoint with the lowest _𝜒_ [2] sum
is chosen for each GAN. This selection criterion, as opposed to selecting the last trained epoch, avoids the
problem of selecting an epoch with an unfavourable fluctuation in the training.


The evolution of this _𝜒_ [2] as a function of the epoch is shown in Figure 9 for pions with 0 _._ 25 _<_ | _𝜂_ | _<_ 0 _._ 3.
The average _𝜒_ [2] decreases with increasing epoch and the fluctuations around the average are typical of
GAN training. The point with the lowest _𝜒_ [2] sum, which in the example presented in Figure 9 occurs at
epoch number 946 000, is the checkpoint used for the simulation of pions in that _𝜂_ range. This procedure is
repeated for all 300 GANs.


**5.2.2 Performance**


The performance of the best epoch for photons with 0 _._ 2 _<_ | _𝜂_ | _<_ 0 _._ 25 is shown in Figure 10. For each of
the 15 energy points, the distribution of the total energy, defined as the sum of the energy in all voxels, is
shown for the Geant4 input samples and the events generated with the GAN. In most cases, the means of
the two distributions are comparable and so are their widths.


Similarly, the performance of the GAN for pions with 0 _._ 2 _<_ | _𝜂_ | _<_ 0 _._ 25 is shown in Figure 11. The first
two energy points show a different shape than the other energy points and are not well described. The
description of the highest energy point is poor due to the difficulties in reproducing the irregular shape and


22


Figure 9: The _𝜒_ [2] sum divided by the number of degree of freedom (NDF) calculated between the reference samples
and the GAN as a function of the number of epochs. The lowest point (in red) represents the selected epoch.


low number of events in the reference sample; given the extreme rarity of such high-energy hadrons in
physics samples, the poor modelling is not of significant concern. Furthermore, in its final configuration
described in Section 7.1, AtlFast3 does not use GANs in this energy range.


The mean and RMS, indicated by the size of the uncertainty bars, of the total energy as a function of the
true particle momentum ( _𝑝_ truth ) is shown in Figures 12(a), (b) and (c) for photons, electrons and pions,
respectively. For photons and electrons, the GANs reproduce the mean energies of the reference samples
except at the low momentum points. The RMS from the GANs is larger than that of the reference sample
for all energies. For pions, the GANs generate distributions with a lower mean and a larger RMS for a
wider energy range.


The total energy, defined as the sum of the energy in all voxels, for particles with momentum 65 GeV as a
function of _𝜂_ for photons, electrons and pions is shown in Figures 13(a)–(c). For photons and electrons, the
GAN and reference sample means agree to better than 1% in almost all the regions, while the distributions
generated by the GANs are wider than the reference samples in the barrel region. Small discrepancies
are observed in the transition regions between detectors, where the energy response is non-Gaussian. For
pions, the means agree to within 4%, with larger discrepancies observed in the barrel region, where the
energy is slightly lower. The FastCaloGAN RMS is larger in both the barrel and endcap regions.


**5.3 Simulation of hits**


The GAN models trained to generate showers in FastCaloGAN are implemented in the ATLAS Athena
software framework using the Lightweight Trained Neural Network (LWTNN) [63].


The kinetic energy _𝐸_ kin of the particle is used as the conditional parameter of the GAN. The output of
the GAN is the energy assigned to each voxel. Each one of these energies must be assigned to a variable


23


Figure 10: Sum of the energy in all voxels for photons with 0 _._ 2 _<_ | _𝜂_ | _<_ 0 _._ 25 . The calorimeter response for Geant4
(solid black line) compared with FastCaloGAN (dashed red line).


number of cells because the voxels in FastCaloGAN can be larger than the ATLAS cells. To assign the
correct amount of energy to each cell, the voxel surface defined in Eq. (2) is sampled uniformly, generating
a grid of hits. Layers that are not binned along the angular direction have their energy uniformly distributed
across the whole annulus surface. The granularity used to sample the voxel is 1 mm in the high-granularity
EMB1 and EME1 layers, while 5 mm is used in the other layers. A maximum of 10 hits are created in
either direction to limit the number of hits that are generated; this is required to have a small simulation
time. The energy generated by the GAN in the voxel is divided uniformly between the hits. The hits are
then assigned to the calorimeter cells using the simplified geometry. The longitudinal mid-position in each
layer is used for the calculation of the hit position.



24


Figure 11: Sum of the energy in all voxels for pions with 0 _._ 2 _<_ | _𝜂_ | _<_ 0 _._ 25 . The calorimeter response for Geant4
(solid black line) is compared with FastCaloGAN (dashed red line).


25


|truth<br>1.2 ATLAS Simulation G4 RMS/p<br>γ 0.20<|η|<0.25<br>FastCaloGAN<br>1.1 and<br>truth<br>1 E〉/p|Col2|Col3|truth<br>1.2 ATLAS Simulation G4 RMS/p<br>e- 0.20<|η|<0.25<br>1.1 FastCaloGAN<br>and<br>1<br>truth<br>0.9 E〉/p|Col5|Col6|
|---|---|---|---|---|---|
|1.4<br>AN/G4<br>0.96<br>0.98<br>1<br>1.02<br> FGAN/G4<br>〉<br>E<br>〈<br>0.7<br>0.8<br>0.9<br>〈|||2<br>AN/G4<br>0.95<br>1<br> FGAN/G4<br>〉<br>E<br>〈<br>0.6<br>0.7<br>0.8<br>〈|||
|1.4<br>AN/G4<br>0.96<br>0.98<br>1<br>1.02<br> FGAN/G4<br>〉<br>E<br>〈<br>0.7<br>0.8<br>0.9<br>〈||||||
|1.4<br>AN/G4<br>0.96<br>0.98<br>1<br>1.02<br> FGAN/G4<br>〉<br>E<br>〈<br>0.7<br>0.8<br>0.9<br>〈||||||
|1.4<br>AN/G4<br>0.96<br>0.98<br>1<br>1.02<br> FGAN/G4<br>〉<br>E<br>〈<br>0.7<br>0.8<br>0.9<br>〈||||||
|1.4<br>AN/G4<br>0.96<br>0.98<br>1<br>1.02<br> FGAN/G4<br>〉<br>E<br>〈<br>0.7<br>0.8<br>0.9<br>〈||||||
|1.4<br>AN/G4<br>0.96<br>0.98<br>1<br>1.02<br> FGAN/G4<br>〉<br>E<br>〈<br>0.7<br>0.8<br>0.9<br>〈||||||
|~~2.5~~<br>~~3~~<br>~~3.5~~<br>~~4~~<br>~~4.5~~<br>~~5~~<br>~~5.5~~<br>~~6~~<br>~~6.5~~/MeV)<br>truth<br>(p<br>10<br>Log<br>0.8<br>1<br>1.2<br>RMS FG|~~2.5~~<br>~~3~~<br>~~3.5~~<br>~~4~~<br>~~4.5~~<br>~~5~~<br>~~5.5~~<br>~~6~~<br>~~6.5~~/MeV)<br>truth<br>(p<br>10<br>Log<br>0.8<br>1<br>1.2<br>RMS FG|~~2.5~~<br>~~3~~<br>~~3.5~~<br>~~4~~<br>~~4.5~~<br>~~5~~<br>~~5.5~~<br>~~6~~<br>~~6.5~~/MeV)<br>truth<br>(p<br>10<br>Log<br>0.8<br>1<br>1.2<br>RMS FG|~~2.5~~<br>~~3~~<br>~~3.5~~<br>~~4~~<br>~~4.5~~<br>~~5~~<br>~~5.5~~<br>~~6~~<br>~~6.5~~/MeV)<br>truth<br>(p<br>10<br>Log<br>1<br>1.5<br>RMS FG|~~2.5~~<br>~~3~~<br>~~3.5~~<br>~~4~~<br>~~4.5~~<br>~~5~~<br>~~5.5~~<br>~~6~~<br>~~6.5~~/MeV)<br>truth<br>(p<br>10<br>Log<br>1<br>1.5<br>RMS FG|~~2.5~~<br>~~3~~<br>~~3.5~~<br>~~4~~<br>~~4.5~~<br>~~5~~<br>~~5.5~~<br>~~6~~<br>~~6.5~~/MeV)<br>truth<br>(p<br>10<br>Log<br>1<br>1.5<br>RMS FG|


(a)





(b)







|1.2 truth<br>ATLAS Simulation G4 RMS/p<br>π± 0.20<|η|<0.25<br>1 FastCaloGAN<br>and<br>0.8 truth<br>〈E〉/p<br>0.6<br>0.4<br>0.2|Col2|Col3|
|---|---|---|
|0.2<br>0.4<br>0.6<br>0.8<br>1<br>1.2<br>truth<br> and RMS/p<br>truth<br>/p<br>〉<br>E<br>〈<br>**_ATLAS_** Simulation<br>|<0.25<br>η<br> 0.20<|<br>±<br>π<br>**G4**<br>**FastCaloGAN**|Simulation<br>|<0.25<br><br>**G4**<br>**FastCal**|Simulation<br>|<0.25<br><br>**G4**<br>**FastCal**|
|1.2<br>1.4<br>AN/G4<br>0.95<br>1<br> FGAN/G4<br>〉<br>E<br>〈|||
|1.2<br>1.4<br>AN/G4<br>0.95<br>1<br> FGAN/G4<br>〉<br>E<br>〈|||
|1.2<br>1.4<br>AN/G4<br>0.95<br>1<br> FGAN/G4<br>〉<br>E<br>〈|||
|~~2.5~~<br>~~3~~<br>~~3.5~~<br>~~4~~<br>~~4.5~~<br>~~5~~<br>~~5.5~~<br>~~6~~<br>~~6.5~~/MeV)<br>truth<br>(p<br>10<br>Log<br>0.8<br>1<br><br>RMS FG|~~2.5~~<br>~~3~~<br>~~3.5~~<br>~~4~~<br>~~4.5~~<br>~~5~~<br>~~5.5~~<br>~~6~~<br>~~6.5~~/MeV)<br>truth<br>(p<br>10<br>Log<br>0.8<br>1<br><br>RMS FG|~~2.5~~<br>~~3~~<br>~~3.5~~<br>~~4~~<br>~~4.5~~<br>~~5~~<br>~~5.5~~<br>~~6~~<br>~~6.5~~/MeV)<br>truth<br>(p<br>10<br>Log<br>0.8<br>1<br><br>RMS FG|


(c)


Figure 12: Sum and RMS of the energy in all voxels normalized to the true momentum for (a) photons, (b) electrons
and (c) pions with 0 _._ 2 _<_ | _𝜂_ | _<_ 0 _._ 25 as a function of the true momentum. The calorimeter response for Geant4 (solid
black line) is compared with FastCaloGAN (dashed red line), which is also abbreviated to FGAN. The uncertainty
bars in the top panel indicate the RMS of the total energy distribution. The ratio of the means of the two energy
distributions is shown in the middle panel, and the ratio of the RMS values is shown in the bottom panel. The error
bars in the ratio indicate its statistical uncertainty. For most points, this uncertainty is smaller than the size of the
markers.


26


|75 ATLAS Simulation G4 [GeV] [GeV]<br>γ E=65.5 GeV<br>FastCaloGAN RMS RMS<br>70<br>and and<br>〈E〉 〈E〉<br>65<br>60<br>55|Col2|75 ATLAS Simulation G4<br>e- E=65.5 GeV<br>FastCaloGAN<br>70<br>65<br>60<br>55|Col4|Col5|Col6|
|---|---|---|---|---|---|
|55<br>60<br>65<br>70<br>75<br> and RMS [GeV]<br>〉<br>E<br>〈<br>**_ATLAS_** Simulation<br> E=65.5 GeV<br>γ<br>**G4**<br>**FastCaloGAN**<br><br> and RMS [GeV]<br>〉<br>E<br>〈||||||
|1.4<br>1.6<br>AN/G4<br>0.985<br>0.99<br>0.995<br>1<br>1.005<br>1.01<br> FGAN/G4<br>〉<br>E<br>〈||1.3<br>1.4<br>1.5<br>.99<br>95<br>1<br>05<br>.01||||
|~~0~~<br>~~1~~<br>~~2~~<br>~~3~~<br>~~4~~<br>~~5~~<br>|<br>η<br>|<br>0.8<br>1<br>1.2<br>RMS FG<br><br><br><br><br>RMS FG|~~0~~<br>~~1~~<br>~~2~~<br>~~3~~<br>~~4~~<br>~~5~~<br>|<br>η<br>|<br>0.8<br>1<br>1.2<br>RMS FG<br><br><br><br><br>RMS FG|~~0~~<br>~~1~~<br>~~2~~<br>~~3~~<br>~~4~~<br>~~5~~<br>|<br>η<br>|<br>0.8<br>0.9<br>1<br>1.1<br>1.2<br>|~~0~~<br>~~1~~<br>~~2~~<br>~~3~~<br>~~4~~<br>~~5~~<br>|<br>η<br>|<br>0.8<br>0.9<br>1<br>1.1<br>1.2<br>|~~0~~<br>~~1~~<br>~~2~~<br>~~3~~<br>~~4~~<br>~~5~~<br>|<br>η<br>|<br>0.8<br>0.9<br>1<br>1.1<br>1.2<br>|~~0~~<br>~~1~~<br>~~2~~<br>~~3~~<br>~~4~~<br>~~5~~<br>|<br>η<br>|<br>0.8<br>0.9<br>1<br>1.1<br>1.2<br>|


(b)







(a)

















(c)


Figure 13: Sum and RMS of the energy in all voxels as a function of | _𝜂_ | for (a) photons, (b) electrons and (c) pions
of momentum 65 GeV . The calorimeter response for Geant4 (solid black line) is compared with FastCaloGAN
(dashed red line), which is also abbreviated to FGAN, while their ratio is shown in the ratio plots. The uncertainty
bars in the top panel indicate the RMS of the total energy distribution. The ratio of the means of the two energy
distributions is shown in the middle panel, and the ratio of the RMS is shown in the bottom panel. The error bars in
the ratio indicate its statistical uncertainty. For most points, this uncertainty is smaller than the size of the markers.


27


#### **6 Simulation of muon punch-through**

Secondary particles created in hadronic showers inside the calorimeter can escape through the back of the
calorimeter and generate hits in the muon spectrometer. This effect is referred to as _muon punch through_ .
These particles are reconstructed in the muon spectrometer and need to be well modelled to accurately
describe the backgrounds of reconstructed muons. A dedicated treatment of these particles is required
because the information about the path of the particles is lost due to the parameterization of the calorimeter
response in AtlFast3. Figure 14 shows the probability of a single pion entering the calorimeter to create
at least one secondary particle which escapes the calorimeter volume with an energy of at least 50 MeV
determined using the Geant4 simulation. The probability increases with increasing momentum _𝑝_ and
varies as a function of _𝜂_ . Particles with energies below 50 MeV are not simulated in the muon spectrometer
because they would have negligible impact.



4194304


2097152


1048576


524288


262144


131072


65536


32768


16384


8192


4096



1


0.8


0.7


0.6


0.5


0.4


0.3


0.2


0.1



0

|Col1|Col2|Col3|A|T|Col6|L|A|S|Col10|Col11|Col12|S|i|m|u|l|a|ti|o|n|Col22|Col23|Col24|Col25|Col26|Col27|Col28|Col29|Col30|Col31|Col32|Col33|Col34|Col35|Col36|Col37|Col38|Col39|Col40|Col41|Col42|Col43|Col44|Col45|Col46|Col47|Col48|Col49|Col50|Col51|Col52|Col53|Col54|Col55|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|||||||||||||S|i|||l||i|o|n|||||||||||||||||||||||||||||||||||
||||||||||||||||||||||||||||||||||||||||||||||||||||||||
||||||||||||||||||||||||||||||||||||||||||||||||||||||||
||||||||||||||||||||||||||||||||||||||||||||||||||||||||
||||||||||||||||||||||||||||||||||||||||||||||||||||||||
||||||||||||||||||||||||||||||||||||||||||||||||||||||||
||||||||||||||||||||||||||||||||||||||||||||||||||||||||
||||||||||||||||||||||||||||||||||||||||||||||||||||||||
||||||||||||||||||||||||||||||||||||||||||||||||||||||||
||||||||||||||||||||||||||||||||||||||||||||||||||||||||
||||||||||||||||||||||||||||||||||||||||||||||||||||||||
||||||||||||||||||||||||||||||||||||||||||||||||||||||||
||||||||||||||||||||||||||||||||||||||||||||||||||||||||
||||||||||||||||||||||||||||||||||||||||||||||||||||||||
||||||||||||||||||||||||||||||||||||||||||||||||||||||||
||||||||||||||||||||||||||||||||||||||||||||||||||||||||
||||||||||||||||||||||||||||||||||||||||||||||||||||||||
||||||||||||||||||||||||||||||||||||||||||||||||||||||||
||||||||||||||||||||||||||||||||||||||||||||||||||||||||
||||||||||||||||||||||||||||||||||||||||||||||||||||||||
||||||||||||||||||||||||||||||||||||||||||||||||||||||||



Primary Pion |η|


Figure 14: The probability of a single-pion event to produce at least one punch-through particle with an energy of at
least 50 MeV as a function of the _𝜂_ and _𝑝_ of the incoming pion determined from Geant4.


The AtlFast3 punch-through parameterization is derived separately for the five types of secondary particles
that can emerge from the back of the calorimeter: photons, electrons, pions, muons, and protons. These
account for 92% of the total punch through. The parameterizations of their multiplicity and kinematics
are determined from single-pion samples simulated using Geant4. As the properties of the secondary
particles depend significantly on the _𝜂_ direction and energy of the incoming pion, the reference samples
within the acceptance of the muon spectrometer | _𝜂_ | ≤ 2 _._ 7 and with momenta between 65 GeV and 4.2 TeV
are used to determine the parameterization. The small number of secondary particles in lower-energy
samples did not allow a parametrization of primary particles with an energy lower than 65 GeV.


The properties of the secondaries described by the parameterization include their energy, and their position
and momentum relative to that of the incoming pion. The position and momentum of the secondaries are
determined via deflection angles, Δ _𝜃_ and Δ _𝜙_, relative to the direction of propagation of the incoming pion.
As an example, Figures 15 and 16 show the histograms extracted from the Geant4 simulation and used
to parameterize the secondaries produced by primary pions with an energy of 524 GeV and | _𝜂_ | ≤ 0 _._ 4.
The peak at 1 GeV is the most probable value of the energy of the secondary pions emerging form the
calorimeter.


28


(a)



(b)



Figure 15: The punch-through probability as a function of the punch-through pion (a) multiplicity and (b) energy.
The error bars indicate the statistical uncertainty and the overflow is not included in the final bins.


During the simulation of AtlFast3, the muon punch-through parameterization is invoked whenever particles
that have some probability of punching through enter the calorimeter. For each incoming particle, the
number of secondaries and their energy, position and momentum are selected randomly from the punchthrough parameterization histograms (see Figures 15 and 16), using them as probability density functions.
The parameterization is interpolated linearly for _𝜂_ and logarithmically for _𝑝_ T to values between the discrete
points used to determine the parameterization. Two sets of correlations are accounted for in the modelling
of the secondaries: the correlations between the relative position and energy and correlations between the
relative momentum and energy. After the multiplicity and properties of the punch-through secondaries
have been determined, their propagation through the muon spectrometer is simulated using Geant4.

#### **7 The combination of FastCaloSim V2 and FastCaloGAN: AtlFast3**


**7.1 Configuration of AtlFast3**


The new fast simulation tool, AtlFast3, is defined by combining the fast simulation tools described above in
a way that balances modelling performance needs with CPU requirements. AtlFast3 uses the Integrated
Simulation Framework (ISF), which allows different simulation tools to be combined in a flexible way [64].
AtlFast3 uses the following configuration as illustrated in Figure 17:


   - Geant4 is used to simulate all particles in the inner detector and muons in all detectors. Hadrons
with kinetic energies below 400 MeV (200 MeV for pions) in the calorimeter are also simulated in

Geant4.


29


(a)



(b)

































(c)



(d)



Figure 16: The punch-through probability as a function of (a) deflection angle in _𝜃_ and energy, (b) deflection angle in
_𝜙_ and energy, (c) relative momentum deflection in _𝜃_ and energy, and (d) relative momentum deflection in _𝜙_ and
energy. Secondary pions with an energy of 524 GeV in the region | _𝜂_ | ≤ 0 _._ 4 from the Geant4 reference samples
were used.


30


- FastCaloSim V2 is used to simulate electrons and photons of all energies and hadrons with kinetic
energies _𝐸_ kin _<_ 8–16 GeV or _𝐸_ kin _>_ 256–512 GeV in the calorimeter. A transition range of energies
is given because the response is interpolated linearly between the two models as discussed later.































**7.2 Configuration of the Fast Calorimeter Simulation**


The configuration of AtlFast3 is determined by comparing the performance of FastCaloSim V2 and
FastCaloGAN.


**7.2.1 Electrons and Photons**


The simulation of electrons and photons relies on the accurate simulation of electromagnetic showers in the
electromagnetic calorimeter. The total reconstructed energy for 65 GeV photons is shown in Figure 18 for
Geant4, FastCaloSim V2 and FastCaloGAN. FastCaloGAN does not model the photon energy correctly
and a similar poor performance is observed for electrons; therefore, FastCaloSim V2 is selected to simulate
all electromagnetic showers. The poor modelling of electromagnetic showers in FastCaloGAN can be
explained as follows: the GANs are trained without the energy resolution correction for the accordion
structure of the calorimeter (see Section 7.4.4), and the energy scale of the detailed Geant4 hits, used in
the training of FastCaloGAN, is slightly lower than that of the full Geant4 hits (see Section 3.1). Both
these effects can be corrected for in future versions of FastCaloGAN but could not be included here due to

time constraints.


31


Figure 18: Reconstructed photon energy for photons generated at the calorimeter surface with an energy of 65 GeV
and 0 _._ 2 _<_ | _𝜂_ | _<_ 0 _._ 25 by Geant4 (solid black line), FastCaloSim V2 (dashed blue line), and FastCaloGAN (dashed
red line). The statistical uncertainties are shown but are similar in size to the points or smaller.


**7.2.2 Low-energy hadrons**



At low energies, the distribution of the average hadron energy response becomes complex and has a
significant dependence on both _𝐸_ kin and | _𝜂_ | as shown in Figure 19(b). This is because the measured energy
depends strongly on the extent to which these shorter showers develop within the active liquid argon
of the electromagnetic calorimeter or within the inactive lead absorbers. As an example, Figure 19(a)
shows the ratio of the average energy response to _𝐸_ kin as a function of _𝐸_ kin for charged pions in the range
0 _._ 20 _<_ | _𝜂_ | _<_ 0 _._ 25. For pions with a kinetic energy of 100 MeV the largest amount of deposited energy is
typically within the liquid argon of the Presampler, which leads to a spike in the energy response. On the
other hand, pions with a kinetic energy of 10 MeV deposit far less energy in the active liquid-argon regions
and more in the inactive regions. In addition, the energy calibration of the Presampler is derived using
high-energy particles, which deposit much less energy in the Presampler, which means that the measured
fraction of shower energy in the Presampler increases further for _𝐸_ kin ≈ 100 MeV.


The dependence of the energy response to low-energy charged pions on _𝜂_ is due to the different amount
of material that the charged pion passes through, which shifts the values of the kinetic energy at which
the spike in the response occurs. Deriving a parameterization for such low-energy hadrons would require
a significantly more complex method for deriving parameterizations in order to achieve high accuracy.
Therefore, in AtlFast3 pions below 200 MeV and all other hadrons below 400 MeV (as shown in Table 6)
are instead simulated by Geant4. Above these energy thresholds their total energy response is modelled
using AtlFast3. This choice does not significantly affect the speed of AtlFast3 because the simulation of
low-energy hadrons requires only a comparatively small amount of CPU time.


Table 6: Hadron energies below which AtlFast3 relies on Geant4 for their simulation


Particle _𝐸_ kin [MeV]

_𝜋_ ~~[±]~~ 200

_𝐾_ [±], _𝐾_ L, _𝑝_ / ¯ _𝑝_, _𝑛_ / _𝑛_ ¯ 400


32


(a)



(b)



Figure 19: Ratio of the average energy response to the generated energy for _𝜋_ [±] for (a) 0 _._ 20 _<_ | _𝜂_ | _<_ 0 _._ 25 and (b)
as a function of | _𝜂_ | and _𝐸_ kin . The error bars indicate the statistical uncertainty of the mean. For most points, this
uncertainty is smaller than the size of the markers.


**7.2.3 Medium-energy hadrons**



For hadronic showers, the number of clusters in a jet plays an important role in modeling the jet substructure
and is therefore used as a metric to compare the performance of FastCaloSim V2 and FastCaloGAN.
Differences in the modelling of the number of clusters between FastCaloSim V2 and FastCaloGAN are
expected because FastCaloGAN can model the correlations within a single event, while FastCaloSim V2
cannot. Figure 20 compares the modelling of the number of clusters in a jet for three different combinations
of FastCaloSim V2 and FastCaloGAN. The hybrid models differ in the energy range over which the
transition between FastCaloSim V2 and FastCaloGAN occurs; for example in the Hybrid 4–8 GeV model,
FastCaloSim V2 is used up to 4 GeV and FastCaloGAN is used above 8 GeV . Between 4 and 8 GeV, the
response is interpolated linearly between the two models as described in Section 7.3. The Hybrid 4–8 GeV
model underestimates the number of constituents, while the Hybrid 16–32 GeV model overestimates the
number of constituents. Therefore, the Hybrid 8-16 GeV model is chosen as the configuration for AtlFast3.
Other key jet variables, including the number of jets, the _𝑝_ T and _𝜂_ distributions and variables used for
substructure, are also checked for these different configurations, which provides additional support for
choosing the Hybrid 8–16 GeV model. Section 8.1.2 discusses the performance of AtlFast3 in modelling
jet variables.


**7.2.4 High-energy hadrons**



At higher energies the modelling of the properties of individual clusters becomes important. Figure 21
compares the number of cells in the calorimeter clusters in Geant4 with FastCaloSim V2 and FastCaloGAN
for pion energies ranging from 65 GeV to 524 GeV. Although FastCaloSim V2 slightly overestimates the
number of cells for all energies, FastCaloGAN significantly underestimates the number of cells and this
becomes more pronounced at higher energy. Studies of additional jet variables, many of which are shown
in Section 8.1.2, confirmed that FastCaloSim V2 has better modelling for higher-energy hadrons. Therefore,
FastCaloSim V2 is used to simulate hadrons with _𝐸_ kin _>_ 256–512 GeV . As shown in Section 8.1.2, despite
these discrepancies, the modelling of higher-level objects such as jets is sufficient for physics analysis.


33


Figure 20: Distribution of the number of constituents in the jets in a 1 _._ 8 _< 𝑝_ T _<_ 2 _._ 5 TeV dijet sample in Geant4
(black triangles) and the combination of FastCaloSim V2 and FastCaloGAN with transitions in the range 4–8 GeV
(blue stars), 8–16 GeV (red diamonds) and 16–32 GeV (green crosses). Here ‘hybrid’ refers to the combination of
FastCaloSim V2 and FastCaloGAN. The statistical uncertainties are shown but may be smaller than the markers.


**7.2.5 Muon punch-through**


The muon punch-through parameterization described in Section 6 is used to simulate particles punching
through the calorimeter. After the multiplicity and properties of the secondaries are determined using the
punch-through parameterization, their path through the muon spectrometer is simulated using Geant4.


**7.3 Energy interpolation**


The FastCaloSim V2 and FastCaloGAN parameterizations are derived using samples with logarithmically
spaced discrete energies, which need to be extrapolated to particles of all energies. In FastCaloSim V2, a
piece-wise third order polynomial spline function is fitted to the total energy response in order to interpolate
to intermediate energies. Furthermore, linear extrapolation is used to reach energies beyond those of
the simulated input samples. The spline interpolations are generated for each particle and each _𝜂_ slice
and are used to rescale the total energy response from the parameterization points. An example of the
energy response and fitted splines for photons and pions in the barrel region is shown in Figure 22. The
energy response for high-energy photons is slightly reduced due to leakage into the Tile calorimeter. In
FastCaloGAN, the conditioning on the particle momentum creates a model that can produce particles of

any energy.


In addition to the interpolation of the total energy response, the other longitudinal and lateral shower
shape properties also need to be interpolated. In FastCaloGAN the shape properties are interpolated
automatically by the GANs, while in FastCaloSim V2 the shape interpolation is done by randomly selecting
the parameterization from the nearest energy point with a probability linear in log( _𝐸_ kin ) and fitted such
that unit probability is reached for the grid energy points.


In the two transition regions between FastCaloSim V2 and FastCaloGAN (for hadrons in the ranges
8–16 GeV and 256–512 GeV ), a spline is used to interpolate between the two models. A smooth energyresponse transition between the two models is obtained since the simulated energies are always scaled to


34


(a)



(b)

























(c)



(d)



Figure 21: Number of cells in the leading cluster for pions in the barrel at different energies in Geant4 (black
triangles), FastCaloSim V2 (red diamonds) and FastCaloGAN (blue stars). The statistical uncertainties are shown but
may be smaller than the markers.


the energy from Geant4. For electrons and photons the spline for the energy response is fitted down to
16 MeV, below which a linear extrapolation is used. For hadrons the energy response is fitted down to a
kinetic energy of 200 MeV, below which Geant4 is used for the simulation.



**7.4 Corrections**


Four different corrections are applied to the calorimeter parameterization in AtlFast3. However, the energy
resolution correction discussed in Section 7.4.1 and the energy _𝜙_ -modulation correction discussed in
Section 7.4.2 are only applied to FastCaloSim V2.


**7.4.1 Energy resolution correction**


The simulation of the resolution of the total energy in FastCaloSim V2 is improved by reweighting the
distribution of simulated energies produced by FastCaloSim V2 to the distribution from Geant4. The ratio


35


(a) Photons





(b) Pions



Figure 22: Energy response, defined as the ratio of the reconstructed energy in the calorimeter cells to the kinetic
energy of the particle, for (a) photons in 1 _._ 05 _<_ | _𝜂_ | _<_ 1 _._ 10 and (b) pions in 0 _._ 20 _<_ | _𝜂_ | _<_ 0 _._ 25 . The red dotted points
represent the response derived at discrete energies, using Geant4 simulated single particles. The black line is a
spline fit used to interpolate between discrete energy points. The statistical uncertainties are shown but are similar in
size to the points or smaller.


of the Geant4 simulated energy to the FastCaloSim V2 simulated energy for each PCA bin is used to
create a pdf. For each simulated total energy the pdf returns an associated probability. During simulation,
for each simulated energy a uniform random number in [0,1] is drawn and if the number is smaller than the
probability obtained from the pdf, the simulated energy is accepted. If the energy is rejected, then the
energy simulation step discussed in Section 4.1 is repeated. The RMS is calculated using at least 99% of all
events and in a ± 3 _𝜎_ range around the mean. This probabilistic reweighting (rw) obtains good agreement
with the Geant4 distribution. Figure 23 shows the resolution for photons, as an example, before and after
the correction, and the RMS of the distribution is indicated.


**7.4.2 Energy** _𝝓_ **-modulation correction**


Due to the accordion structure of the EM calorimeter, the total deposited energy is modulated in the
_𝜙_ -direction as shown in Figure 24(a), where | _𝜙_ mod | = |mod( _𝜙_ calo _, 𝜋_ / 512 )| . The calibration applied during
the ATLAS electron and photon reconstruction makes a correction for the _𝜙_ -modulation in the energy
response observed in Geant4; this calibration impacts the resolution of the reconstructed energy. The
modulation is not reproduced in FastCaloSim V2, because it does not have a functional dependence on _𝜙_ .
The resolution of showers in the electromagnetic calorimeter produced by FastCaloSim V2 is corrected
by deriving the energy parameterization of Section 4.1 after removing the modulation of the energy
as a function of _𝜙_ in the reference samples. This procedure is applied to particles with energies of at
least 16 GeV ; below this threshold the effect is negligible and can be ignored. Figure 24(b) shows the
energy response from Geant4 for photons with and without the removal of the _𝜙_ modulation compared
with the prediction from FastCaloSim V2. Good agreement in the modelling of the resolution between
FastCaloSim V2 and Geant4 is obtained for the Geant4 samples with the _𝜙_ -modulation removed. As a


36


Figure 23: The simulated total energy before (blue stars) and after (red diamonds) probabilistic reweighting for a
photon of energy 262 GeV in the range 0 _._ 4 _<_ | _𝜂_ | _<_ 0 _._ 45 compared with Geant4 (black triangles). The RMS of each
distribution is indicated in the legend. The statistical uncertainties are shown but may be smaller than the markers.





















(a)



(b)



Figure 24: (a) The total energy response exhibits a dependence on the impact position in _𝜙_ of the particle in the
calorimeter cell ( | _𝜙_ mod | ), shown for 65 GeV photons with 0 _._ 2 _<_ | _𝜂_ | _<_ 0 _._ 25 (Geant4). The ratio has been shifted
such that mean ratio of the energy from Geant4 to the true energy is unity. (b) The impact of the correction on
Geant4 simulation (gray triangles are without correction, black are with corrections) and the result of the stand-alone
simulation for 131 GeV photons with 1 _._ 65 _<_ | _𝜂_ | _<_ 1 _._ 7 to which the correction has been applied as well as the
reweighting described in Section 7.4.1. The statistical uncertainties are shown in the error bars.


consequence of this strategy, during the reconstruction of electrons and photons simulated with AtlFast3,
a set of calibrations without a correction for the energy modulation in the _𝜙_ -direction must be applied,
differing from the calibrations used for full simulation samples. This procedure particularly improves the
modelling of the resolution of the calibrated energies for photons and electrons in AtlFast3.



**7.4.3 Hadron total energy correction**


The hadron total energy correction accounts for the difference between the charged-pion response, which is
used to derive the calorimeter parameterizations, and the response to other hadron species. It is particularly



37


Figure 25: Energy response correction factors as a function of the true kinetic energy for protons, neutrons and
kaons (left) in the barrel and their antiparticles (right). The kinetic energy for antiparticles includes their mass. The
coloured bands indicate the size of the statistical uncertainty in the correction.


important at low energies, where the kinetic energy of a hadron is close to its mass.


The hadron total energy correction is derived using simulated samples of (anti)protons, (anti)neutrons and
(anti)kaons as described in Section 3. Using Geant4, the parameterized energy is corrected by the ratio of
the mean simulated hadron energy response, ⟨ _𝐸_ G4 [h] [⟩] [, to the mean simulated pion energy response,] [ ⟨] _[𝐸]_ G4 _[𝜋]_ [⟩] [. A]
further rescaling must be applied because the reference samples were generated using the momentum of the
particle while the _𝐸_ kin is used for the parameterization. This is achieved by calculating the pion-to-hadron
ratio of kinetic energies for each true momentum in the reference samples. During AtlFast3 simulation,
hadrons are then simulated using the charged-pion parameterization that provides the total energy _𝐸_ Total
given the kinetic energy of the pion, _𝐸_ kin _[𝜋,]_ [true], but with an additional correction based on the kinetic energy
of the hadron, _𝐸_ kin [h,true] . The corrected energy response is then given by:



⟨ _𝐸_ G4 [h] [⟩] _𝐸_ kin _[𝜋,]_ [true]
_𝐸_ Total [corr h] [=] ⟨ _𝐸_ G4 _[𝜋]_ [⟩×] _𝐸_ kin [h,true] × _𝐸_ Total



The value _𝐸_ Total [corr h] [is the corrected energy. The hadron total energy correction is linearly interpolated between]
the logarithmically spaced energy grid points. Figure 25 shows an example of the factor applied for the
hadron total energy correction as a function of the true _𝐸_ kin for protons, neutrons and kaons. The hadron
total energy correction is largest at small kinetic energies and decreases with increasing energy. It does not
depend strongly on _𝜂_ and is similar for protons and neutrons.


**7.4.4 Residual energy response correction**


The residual energy response correction is applied to correct the total energy response for electrons, photons
and pions from the parameterizations to match the average response of Geant4 after the full ATLAS


38


Figure 26: Residual energy response correction factors as a function of the true kinetic energy for photons, electrons
and pions in the endcap. The coloured bands indicate the size of the statistical uncertainty in the correction.


simulation and reconstruction chain. This correction can, therefore, correct for differences introduced
during digitization and reconstruction. The residual energy response correction is the ratio of the average
reconstructed energy when using Geant4, ⟨ _𝐸_ G4 ⟩ to the average reconstructed energy from AtlFast3,
⟨ _𝐸_ AF3 ⟩. The residual energy response correction is calculated and applied as follows:


_𝐸_ Total [corr res] (p) = ⟨ _𝐸_ G4 (p)⟩/⟨ _𝐸_ AF3 (p)⟩× _𝐸_ Total (p)


where p = [ _𝑒, 𝛾, 𝜋_ ].


It is derived for each parameterization grid point and linearly interpolated between the simulated energy
points. Figure 26 shows an example of the derived residual energy response correction as a function of
the true _𝐸_ kin for photons, electrons and pions. The residual energy response correction is at the per-mil
level for electrons and photons and only slightly larger for pions and hence only applied when statistically
significant.


**7.4.5 Simplified geometry shower shape correction**


The hits generated by FastCaloSim V2 or FastCaloGAN are assigned to calorimeter cells using a simplified
cuboid geometry. This introduces a bias in the energy distribution, which can result in a significant number
of hits being assigned to neighbouring calorimeter cells. To account for this effect while maintaining the
reduced simulation CPU time afforded by the simplified geometry, a small displacement in _𝜙_ is assigned
to each hit before geometrically matching it to a cell with the simplified geometry. This procedure is
substantially easier and faster than geometrically matching a hit to the cells in the complex ATLAS
liquid-argon accordion structure.


A pdf is derived from the difference between the cell assignment probabilities calculated in Geant4 and
FastCaloSim V2. The correction is made using the pdf to randomly assign a displacement in _𝜙_ to a hit.
Figure 27 shows the bias in deposited energy in each cuboid before and after this correction has been
applied. Good agreement in the cell energy between AtlFast3 and Geant4 is observed once this correction
has been applied.


39


_**ATLAS**_ Simulation No correction


0.1


0.05


0


−0.05



1.4


1.2


1


0.8


0.6


0.4



_**ATLAS**_ Simulation Corrected


0.1


0.05


0


−0.05



−0.− ~~1~~ 0.1 −0.05 0 0.05 0.1

|Col1|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|Col10|Col11|Col12|Col13|Col14|Col15|Col16|Col17|Col18|Col19|Col20|Col21|Col22|Col23|Col24|Col25|Col26|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|||||||||||||||||||||||||||
|||||||||||||||||||||||||||
|||||||||||||||||||||||||||
|||||||||||||||||||||||||||
|||||||||||||||||||||||||||
|||||||||||||||||||||||||||
|||||||||||||||||||||||||||
|||||||||||||||||||||||||||
|||||||||||||||||||||||||||
|||||||||||||||||||||||||||
|||||||||||||||||||||||||||
|||||||||||||||||||||||||||
|||||||||||||||||||||||||||
|||||||||||||||||||||||||||
|||||||||||||||||||||||||||
|||||||||||||||||||||||||||
|||||||||||||||||||||||||||
|||||||||||||||||||||||||||
|||||||||||||||||||||||||||



∆η(particle, cell)


(a) before correction



−0.− ~~1~~ 0.1 −0.05 0 0.05 0.1

|Col1|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|Col10|Col11|Col12|Col13|Col14|Col15|Col16|Col17|Col18|Col19|Col20|Col21|Col22|Col23|Col24|Col25|Col26|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|||||||||||||||||||||||||||
|||||||||||||||||||||||||||
|||||||||||||||||||||||||||
|||||||||||||||||||||||||||
|||||||||||||||||||||||||||
|||||||||||||||||||||||||||
|||||||||||||||||||||||||||
|||||||||||||||||||||||||||
|||||||||||||||||||||||||||
|||||||||||||||||||||||||||
|||||||||||||||||||||||||||
|||||||||||||||||||||||||||
|||||||||||||||||||||||||||
|||||||||||||||||||||||||||
|||||||||||||||||||||||||||
|||||||||||||||||||||||||||
|||||||||||||||||||||||||||
|||||||||||||||||||||||||||
|||||||||||||||||||||||||||



∆η(particle, cell)


(b) after correction



1.5


1.4


1.3


1.2


1.1


1


0.9


0.8


0.7


0.6


0.5



Figure 27: The ratio of the energies assigned to each cuboid of the second electromagnetic barrel layer in AtlFast3
and Geant4 for a photon of 65 GeV in the range 0 _._ 20 _<_ | _𝜂_ | _<_ 0 _._ 25 using a simplified cuboid geometry and after
applying the correction for the simplified geometry.

#### **8 Performance of AtlFast3**


The performance of AtlFast3 is studied by comparing the modeling of reconstructed quantities (Section 8.1)
and important kinematic variables from physics analyses (Section 8.2) in AtlFast3, Geant4, and AtlFastII.
The reconstructed objects that depend on the performance of the calorimeter are electrons and photons,
jets, and _𝜏_ -leptons. For _𝑏_ -tagging and for other particles, such as muons, the performance of AtlFast3
depends primarily on the performance of the tracking detectors. This is studied as part of the validation
of AtlFast3, and we focus the specific case of muon punch-through in this paper because no significant
differences from Geant4 are observed. Section 8.3 discusses the CPU performance of AtlFast3.


**8.1 Performance of AtlFast3 on objects for physics analysis**


**8.1.1 Reconstructed photons and electrons**


Electron and photon candidates are reconstructed from topological clusters of deposited energy in the
electromagnetic calorimeter, and in the case of electrons, tracks in the inner detector are matched to the
clusters [42]. For physics analysis, identification criteria are defined by requirements on shower shape and
track quality. These identification criteria are labelled as ‘loose’, ‘medium’, and ‘tight’. The identification
of electrons uses information from the inner detector, so a single electron sample with uniform _𝑝_ T, _𝜂_ and
_𝜙_ -coverage are generated at the center of the ATLAS detector for validation in this section. These samples
are then simulated with pileup overlaid. The efficiencies for both electrons and photons are validated on
an inclusive _𝜂_ and _𝑝_ T range that extends beyond what is typically considered for performance studies.
Figure 28(a) compares the electron identification efficiencies for the ‘tight’ criterion as a function of the
reconstructed _𝑝_ T for AtlFastII, AtlFast3 and Geant4. While AtlFastII agrees with Geant4 to better than
5% in the electron _𝑝_ T range from 30 GeV to 300 GeV, AtlFast3 agrees with Geant4 to within 2% in most
of the phase space. The ‘tight’ photon identification efficiency is shown in Figure 28(b) as a function of the
photon _𝑝_ T for AtlFastII, AtlFast3 and Geant4. Except at very low _𝑝_ T, AtlFast3 agrees with Geant4 to
within a few percent with better modelling than AtlFastII.


40


|Efficiency<br>ATLAS Simulation<br>1<br>0.8 Identification<br>0.6<br>G4<br>0.4 Single electrons AF2<br>s=13 TeV<br>AF3<br>Tight criteria<br>0.2<br>0<br>1.04 3 G4<br>1.02<br>1 F /|ATLAS Simulation Efficiency<br>1<br>H→γγ s=13 TeV<br>0.8 Identification<br>Tight criteria<br>0.6<br>G4<br>0.4 AF2<br>AF3<br>0.2<br>0<br>1.05 G4<br>1 /<br>F|
|---|---|
|[GeV]<br>T<br>electron p<br>~~30 40~~<br>~~100~~<br>~~200 300~~<br>~~1000~~<br>~~2000~~<br>A<br>0.92<br>0.94<br>0.96<br>0.98<br>|[GeV]<br>T<br>photon p<br>~~10~~<br>~~100~~<br>A<br>0.85<br>0.9<br>0.95|


(b)







(a)







Figure 28: ‘Tight’ identification efficiencies for single electrons with true energy greater than 20 GeV (a) and photons
from _𝐻_ → _𝛾𝛾_ decays (b) inclusive in | _𝜂_ | _<_ 2 _._ 5 as a function of their reconstructed _𝑝_ T for Geant4 (black triangles),
AtlFastII (blue stars), and AtlFast3 (red diamonds). The statistical uncertainties are shown but may be smaller than
the size of the markers.


**8.1.2 Reconstructed jets**


Jets are reconstructed using a variety of reconstruction algorithms and inputs from the calorimeters and the
inner detector. In addition, different pileup mitigation and jet grooming algorithms are applied.


Particle-flow (EMPFlow) jets, which are constructed using EM-scale topological clusters [45], are
reconstructed with the anti- _𝑘_ _𝑡_ algorithm [44, 65] with a radial distance parameter _𝑅_ = 0 _._ 4, using charged
constituents associated with the primary vertex [41] and neutral particle-flow constituents as inputs [45].
Large-radius jets ( _𝑅_ = 1 _._ 0) are reconstructed by applying the anti- _𝑘_ _𝑡_ algorithm to locally calibrated
topological clusters (LCTopo) [49] and the newer alternative of Unified Flow Objects (UFO) [66].


The performance of AtlFast3 with EMPFlow jets is assessed using the _𝑝_ T of the leading jet and the
pseudorapidity distribution of the sub-leading jet in a _𝑡𝑡_ ¯ sample, which are shown in Figure 29. For both
distributions, AtlFastII and AtlFast3 are consistent with Geant4 at the percent level. In the forward _𝜂_
regions of Figure 29(b), AtlFast3 shows better agreement than AtlFastII with Geant4 thanks to the updated
parameterization in the forward region of the detector.


For higher- _𝑝_ T jets, the simulation of the detailed structure within the jet plays an important role in the
efficiency and classification. To provide better coverage for higher jet _𝑝_ T, the _𝑍_ [′] and _𝑊_ [′] boson samples were
reweighted to have a flat leading-jet _𝑝_ T spectrum as described in Section 3.3. Figure 30 shows the number
of charged constituents for leading jets with _𝑝_ T _>_ 200 GeV from a sample containing _𝑍_ [′] → _𝑡𝑡_ ¯ events.
Figure 30(a) shows EMPFlow jets reconstructed with the anti- _𝑘_ _𝑡_ algorithm with a radius parameter _𝑅_ = 0 _._ 4,
while Figure 30(b) shows UFO jets reconstructed with the anti- _𝑘_ _𝑡_ algorithm with _𝑅_ = 1 _._ 0. The number of
constituents in the EMPFlow jets is significantly underestimated by AtlFastII, while AtlFast3 reproduces
the distribution from Geant4 within statistical uncertainties for jets with more than 14 constituents. For


41


|0.08 Normalized<br>ATLAS Simulation<br>0.07 s=13 TeV, tt<br>Jet p>20 GeV, EMPFlow R=0.4 jets<br>0.06 T G4<br>0.05 AF2<br>0.04 AF3<br>Unit<br>0.03|0.2<br>Normalized<br>0.18 G4 ATLAS Simulation<br>0.16 AF2 Js e= t 1 p3 > T 2e 0V G, t et V, EMPFlow R=0.4 jets<br>0.14 AF3 T<br>0.12<br>0.1<br>Unit<br>0.08<br>0.06|
|---|---|
|0.01<br>0.02<br>2~~0~~<br>~~40~~<br>~~60~~<br>~~80~~<br>~~100~~<br>~~120~~<br>~~140~~<br>~~160~~<br>~~180~~<br>~~20~~0<br>[GeV]<br>T<br>Leading Jet p<br>0.8<br>1<br>1.2<br>AF/G4|0.02<br>0.04<br><br>~~4~~<br>~~−~~<br>~~3~~<br>~~−~~<br>~~2~~<br>~~−~~<br>~~1~~<br>~~−~~<br>~~0~~<br>~~1~~<br>~~2~~<br>~~3~~<br>~~4~~<br>η<br>Sub-leading Jet<br>0.8<br>1<br>1.2<br>AF/G4|
|||


(a)



(b)



Figure 29: The transverse momentum distribution of the leading jets (a) and the pseudorapidity distribution of the
sub-leading jets (b) in a _𝑡𝑡_ ¯ sample with Geant4 (black triangles), AtlFastII (blue stars), and AtlFast3 (red diamonds).
The jets are EMPFlow jets with _𝑅_ = 0 _._ 4 . The statistical uncertainties are shown but may be smaller than the size of
the markers.


events with fewer constituents, AtlFast3 slightly underestimates the number of constituents. For the UFO
jets, agreement with Geant4 improves significantly, going from a 20% difference in AtlFastII to less than
10% in AtlFast3.















|0.18<br>Normalized<br>0.16 ATLAS Simulation<br>s=13 TeV, Z'(4 TeV)→tt<br>0.14 Jet p T>20 GeV, EMPFlow R=0.4 jets<br>0.12<br>0.1<br>0.08 Unit<br>G4<br>0.06<br>AF2|0.1<br>Normalized<br>0.09 ATLAS Simulation<br>s=13 TeV, Z'(4 TeV)→tt<br>0.08 Jet p > 200 GeV, UFO R=1.0 jets<br>0.07 T<br>0.06 G4<br>0.05 AF2<br>0.04 AF3 Unit<br>0.03|
|---|---|
|0.02<br>0.04<br><br>AF3<br>~~0~~<br>~~10~~<br>~~20~~<br>~~30~~<br>~~40~~<br>~~50~~<br>~~60~~<br>~~70~~<br>Leading Jet Number of Constituents<br>0.6<br>0.8<br>1<br>1.2<br>1.4<br>AF/G4|0.01<br>0.02<br><br><br>~~0~~<br>~~20~~<br>~~40~~<br>~~60~~<br>~~80~~<br>~~100~~<br>~~12~~0<br>Leading Jet Number of Constituents<br>0.6<br>0.81<br>1.2<br>1.4<br>AF/G4|
|||


(a)



(b)



Figure 30: Distribution of the number of constituents in the leading jets for EMPFlow jets with _𝑅_ = 0 _._ 4 (a) and
UFO jets with _𝑅_ = 1 _._ 0 (b) in the _𝑍_ [′] sample in Geant4 (black triangles), AtlFastII (blue stars), and AtlFast3 (red
diamonds). The statistical uncertainties are shown but may be smaller than the size of the markers.


Variables commonly used in jet-tagging algorithms include the energy-correlation-function ratio, _𝐷_ 2, for
two-body decays and the _𝑛_ -subjettiness ratio, _𝜏_ 32, for three-body decays [67, 68]. Figure 31 shows the _𝐷_ 2
variable reconstructed using the UFO algorithm with Geant4, AtlFastII, and AtlFast3 on a _𝑊_ [′] sample.
AtlFast3 significantly improves the modelling of _𝐷_ 2, particularly at lower values.


Figure 32 shows _𝜏_ 32 for different large-radius jet algorithms. For the UFO jets in Figure 32(a), AtlFastII
reproduces the distribution of Geant4 to within 20% and AtlFast3 improves this further to within 10%. For
the LCTopo jets shown in Figure 32(b) the modelling from AtlFastII is poor but is significantly improved
with AtlFast3, which obtains agreement to within 20%. The improvement for LCTopo is expected to be
larger than for UFO because UFO includes tracking information.


42


Figure 31: The _𝐷_ 2 variable for the leading jets in a _𝑊_ [′] sample reconstructed using the UFO algorithm with radius
parameter _𝑅_ = 1 _._ 0 with Geant4 (black triangles), AtlFastII (blue stars), and AtlFast3 (red dimaonds). The statistical
uncertainties are shown but may be smaller than the size of the markers.













|0.06<br>Normalized<br>ATLAS Simulation<br>0.05 s=13 TeV, Z'(4 TeV)→tt<br>Jet p>200 GeV, UFO R=1.0 jets<br>T G4<br>0.04<br>AF2<br>0.03 AF3<br>Unit<br>0.02|0.06<br>Normalized<br>ATLAS Simulation<br>0.05 s=13 TeV, Z'(4 TeV)→tt<br>Jet p>200 GeV, LCTopo R=1.0 jets<br>T G4<br>0.04<br>AF2<br>0.03 AF3<br>Unit<br>0.02|
|---|---|
|0.01<br>~~0~~<br>~~0.1~~<br>~~0.2~~<br>~~0.3~~<br>~~0.4~~<br>~~0.5~~<br>~~0.6~~<br>~~0.7~~<br>~~0.8~~<br>~~0.9~~<br>~~1~~<br>32<br>τ<br>Leading jet<br>0.8<br>1<br>1.2<br>AF/G4|0.01<br>~~0~~<br>~~0.1~~<br>~~0.2~~<br>~~0.3~~<br>~~0.4~~<br>~~0.5~~<br>~~0.6~~<br>~~0.7~~<br>~~0.8~~<br>~~0.9~~<br>~~1~~<br>32<br>τ<br>Leading jet<br>0.8<br>1<br>1.2<br>AF/G4|
|||


(a)



(b)



Figure 32: The _𝜏_ 32 variable for the leading jets in a _𝑍_ [′] sample reconstructed using the UFO algorithm with radius
parameter _𝑅_ = 1 _._ 0 (a) and the LCTopo algorithm (b) with Geant4 (black triangles), AtlFastII (blue stars), and
AtlFast3 (red diamonds). The statistical uncertainties are shown but may be smaller than the size of the markers, and
the dark blue arrows indicate that a point is beyond the _𝑦_ -axis range.


**8.1.3 Reconstructed hadronic** _𝝉_ **-lepton decays**


Hadronically decaying _𝜏_ -leptons are reconstructed in the ATLAS detector using their decays to one or
three charged hadrons along with neutral particles [69–71]. The decays are labelled by the number (Y)
of charged particles and the number (X) of neutral particles, YpXn. The _𝜏_ reconstruction algorithm is
seeded by the presence of a reconstructed jet. Figure 33 compares the number of events in different _𝜏_
decay topologies identified in a _𝑍_ _[★]_ / _𝛾_ _[★]_ → _𝜏𝜏_ Drell–Yan (DY) sample, filtered for an off-shell mass of
2.0–2.25 TeV, for Geant4, AtlFastII and AtlFast3 using _𝜏_ -candidates with _𝑝_ T _>_ 10 GeV and | _𝜂_ | _<_ 2 _._ 5. For
all cases, except 1pXn, both AtlFastII and AtlFast3 agree with Geant4 to better than 10% for reconstructed
_𝜏_ matched to a true _𝜏_ and better than 5% for reconstructed _𝜏_ ummatched to a true _𝜏_ (i.e. for fake _𝜏_ -leptons).
The 1pXn case has more neutral calorimeter clusters, and the improved lateral correlations of calorimeter
clusters resulted in the better agreement of AF3 with G4. The performance of AtlFastII and AtlFast3 is
similar, with slightly better performance in AtlFastII for true _𝜏_ -leptons and slightly better performance in
AtlFast3 for fake _𝜏_ -leptons.


43


|0.7 Normalized<br>ATLAS Simulation<br>G4<br>0.6 s = 13 TeV, Z(DY)→ττ<br>true τ AF2<br>0.5 AF3<br>0.4<br>Unit<br>0.3<br>0.2<br>0.1|Normalized<br>ATLAS Simulation<br>0.4 s = 13 TeV, Z(DY)→ττ G4<br>fake τ AF2<br>AF3<br>0.3<br>Unit<br>0.2<br>0.1|
|---|---|
|~~1p0n~~<br>~~1p1n~~<br>~~1pXn~~<br>~~3p0n~~<br>~~3pXn~~<br> Tau decay modes<br>0.8<br>1<br>1.2<br>AF/G4|~~1p0n~~<br>~~1p1n~~<br>~~1pXn~~<br>~~3p0n~~<br>~~3pXn~~<br> Tau decay modes<br>0.9<br>1<br>1.~~1~~<br>AF/G4|


(a)



(b)



Figure 33: Hadronic _𝜏_ -lepton decay modes for reconstructed _𝜏_ -leptons matched to true _𝜏_ -leptons (a) and reconstructed
_𝜏_ -leptons not matched to true _𝜏_ -leptons (b) in a _𝑍_ _[★]_ / _𝛾_ _[★]_ → _𝜏𝜏_ Drell–Yan sample filtered for an off-shell mass of
2.0–2.25 TeV . The decays with one or three charged-particle tracks are denoted by 1p and 3p respectively. X (= 1 _,_ 2 _,_ 3)
denotes the number of neutral particles. The statistical uncertainties are shown but may be smaller than the size of
the markers.


Accurate modelling of the structure of the constituents within _𝜏_ -jets can be challenging for fast simulation but
is crucial in obtaining an accurate simulation of _𝜏_ candidates. Figure 34 compares the numbers of simulated
clusters within true (left) and fake (right) _𝜏_ candidates. In both cases, AtlFastII significantly underestimates
the number of clusters, while AtlFast3 is consistent with Geant4 within statistical uncertainties.


**8.1.4 Reconstructed muons**


Muons are reconstructed from tracks in the muon spectrometer matched to tracks in the inner detector.
The _𝑝_ T distributions of all reconstructed muons from Geant4, AtlFastII and AtlFast3 _𝑍_ → _𝜇𝜇_ samples
are compared in Figure 35(a). Both AtlFastII and AtlFast3 reproduce the _𝑝_ T spectrum from Geant4.
Figure 35(b) compares the number of muon candidates passing the different muon reconstruction working
points. Both AtlFastII and AtlFast3 agree with Geant4 within uncertainties as expected because prompt
muons are almost exclusively simulated with Geant4 for all three samples.


The performance of the muon punch-through simulation is validated by comparing misidentified muon
candidates from hadronic activity produced in fully simulated Geant4 events with those produced by
AtlFast3. Figure 36 compares the reconstructed _𝑝_ T of fake muons created by 500 GeV single pions
(inclusive in _𝜂_ ) between Geant4 and AtlFast3. As muon punch through is not simulated in AtlFastII,
only Geant4 and AtlFast3 are shown. Agreement to better than 20% is observed in most parts of the
distributions.


The number of muon segments in jets reconstructed in the muon spectrometer is shown in Figure 37. A


44


|Normalized<br>0.14 ATLAS Simulation<br>G4<br>s = 13 TeV, Z(DY)→ττ<br>0.12 true 1-prong τ AF2<br>0.1 AF3<br>0.08<br>Unit<br>0.06<br>0.04<br>0.02|Normalized<br>0.12 ATLAS Simulation<br>G4<br>s = 13 TeV, Z(DY)→ττ<br>0.1 fake 1-prong τ AF2<br>AF3<br>0.08<br>0.06 Unit<br>0.04<br>0.02|
|---|---|
|~~0~~<br>~~5~~<br>~~10~~<br>~~15~~<br>~~20~~<br>~~25~~<br>~~30~~<br>cluster<br> N<br>0.5<br>1<br>1.5<br>AF/G4|~~0~~<br>~~5~~<br>~~10~~<br>~~15~~<br>~~20~~<br>~~25~~<br>~~30~~<br>cluster<br> N<br>0.5<br>1<br>1.5<br>AF/G4|



Figure 34: Number of clusters in hadronic _𝜏_ -decay candidates reconstructed with one charged track (1p) and either
matched (a) or not matched (b) to a true _𝜏_ -lepton in an _𝑍_ _[★]_ / _𝛾_ _[★]_ → _𝜏𝜏_ Drell–Yan sample filtered for an off-shell mass
of 2.0–2.25 TeV. The statistical uncertainties are shown but may be smaller than the size of the markers.


_𝑍_ [′] → _𝑡𝑡_ ¯ sample is used because it includes prompt muons from the (anti-)top quark decays and particles
produced by jets punching through the calorimeter. AtlFastII underestimates the number of muon segments,
while AtlFast3 shows better agreement with Geant4. In particular, AtlFastII reproduces the number of
muon segments only up to three while AtlFast3 reproduces the number of muon segments up to seven.


**8.1.5 Reconstructed** _𝑬_ **[miss]**
**T**


The missing transverse momentum ( _𝐸_ T [miss] ) [46] is the negative vector sum of the reconstructed momenta
of EMPFlow jets, electrons, photons, _𝜏_ -leptons, and muons, plus any other tracks associated with the
hard-scatter primary vertex, and is used to look for transverse momentum imbalance in _𝑝𝑝_ collisions. The
performance of the _𝐸_ T [miss] reconstruction is therefore sensitive to the modelling of all reconstructed objects.
Figurein _𝑡𝑡_ ¯ events. Both AtlFastII and AtlFast3 reproduce the 38 shows the difference between the true _𝐸_ T [miss] _𝐸_ and the reconstructed T [miss] distribution from Geant4 within the statistical _𝐸_ T [miss] in the _𝑥_ and _𝑦_ directions
uncertainties. Moreover, no significant differences between AtlFastII and AtlFast3 are observed, and this is
attributed to their good agreement in the jet _𝑝_ T shown in Figure 29(a).


45


|Normalized<br>ATLAS Simulation<br>0.07<br>s = 13 TeV, Z → µµ<br>0.06 G4<br>AF2<br>0.05<br>Unit<br>AF3<br>0.04<br>0.03<br>0.02<br>0.01|Normalized<br>1 A T L A S S im u la tio n<br>s = 13 TeV, Z → µµ<br>0.8<br>G4<br>AF2 0.6 Unit<br>AF3<br>0.4<br>0.2<br>0|
|---|---|
|~~0~~<br>~~10~~<br>~~20~~<br>~~30~~<br>~~40~~<br>~~50~~<br>~~60~~<br>~~70~~<br>~~80~~<br> [GeV]<br>T<br>Reconstructed muon p<br>0.8<br>0.9<br>1<br>1.1<br>1.2<br>AF/G4|~~Tight~~<br>~~Medium~~<br>~~Loose~~<br>~~VeryLoose~~<br>Muon identification efficiency<br>0.8<br>0.9<br>1<br>1.1<br>1.2<br>AF/G4|


(b)









|ATLAS Simulation<br>s = 13 TeV, Z → µµ<br>G<br>A<br>A|Col2|
|---|---|
|||
|||
|||
|||
|||


(a)





Figure 35: The (a) reconstructed muon transverse momentum distribution and (b) identification efficiency for different
muon working points for a _𝑍_ → _𝜇𝜇_ sample generated with _𝑝_ T ( _𝑍_ ) = 0 for Geant4, AtlFastII, and AtlFast3. The
statistical uncertainties are shown but may be smaller than the size of the markers.








|ATLAS Simulation<br>500 GeV, π±, |η| ≤ 3.00<br>G4<br>AF3|Col2|Col3|Col4|Col5|Col6|Col7|
|---|---|---|---|---|---|---|
||||||||
||||||||
||||||||
||||||||
||||||||
||||||||



Figure 36: Comparison of muon punch-through simulation in AtlFast3 and Geant4 as a function of the _𝑝_ T of
misidentified muons from 500 GeV single-pion events. The statistical uncertainties are shown but may be smaller
than the size of the markers.


46


Figure 37: Comparison of muon segments in jets reconstructed with a radius parameter of 0.4 using the EMPFlow
algorithm in a _𝑍_ [′] → _𝑡𝑡_ ¯ sample with a _𝑍_ [′] mass of 4 TeV in Geant4 (black triangles), AtlFastII (blue stars), and
AtlFast3 (red diamonds). The statistical uncertainties are shown but may be smaller than the size of the markers.
















|1 Normalized<br>ATLAS Simulation<br>s=13 TeV, tt<br>EMPFlow R=0.4 jets<br>10−1 G4<br>AF2<br>10−2 AF3<br>Unit<br>10−3<br>1.2 G4|1 Normalized<br>ATLAS Simulation<br>s=13 TeV, tt<br>EMPFlow R=0.4 jets<br>10−1 G4<br>AF2<br>10−2 AF3<br>Unit<br>10−3<br>1.2 G4|
|---|---|
|~~100~~<br>~~−~~<br>~~50~~<br>~~−~~<br>~~0~~<br>~~50~~<br>~~100~~<br> [GeV]<br>miss<br>T,Truth,x<br>_E_<br> -<br>miss<br>T,x<br>_E_<br>0.8<br>1<br>AF/|~~100~~<br>~~−~~<br>~~50~~<br>~~−~~<br>~~0~~<br>~~50~~<br>~~100~~<br> [GeV]<br>miss<br>T,Truth,y<br>_E_<br> -<br>miss<br>T,y<br>_E_<br>0.8<br>1<br>AF/|
|||



Figure 38: The difference between the true _𝐸_ T [miss] and the reconstructed _𝐸_ T [miss] in the _𝑥_ (a) and _𝑦_ (b) directions for a _𝑡𝑡_ ¯
sample for Geant4 (black triangles), AtlFastII (blue stars), and AtlFast3 (red diamonds). The statistical uncertainties
are shown but may be smaller than the size of the markers.


47


**8.2 Performance of AtlFast3 in physics analysis**


The performance of AtlFast3 for physics analysis is studied using reconstructed particle masses from
selected physics analyses.


The reconstructed mass of Higgs bosons decaying into two photons is used to further evaluate the
performance for photons. Events are selected by requiring two photons with _𝑝_ T _>_ 0 _._ 35 _𝑚_ _𝛾𝛾_ and
_𝑝_ T _>_ 0 _._ 25 _𝑚_ _𝛾𝛾_, and with | _𝜂_ | _<_ 1 _._ 37 or 1 _._ 52 _<_ | _𝜂_ | _<_ 2 _._ 47. A comparison of Geant4, AtlFastII and
AtlFast3 is shown in Figure 39. Both AtlFastII and AtlFast3 reproduce the mean of the distribution with
high accuracy. AtlFast3 has better modelling of the width and agrees with Geant4 to within 5%, while
AtlFastII overestimates the width of the distribution by 10%.


Events containing Drell–Yan processes are used extensively in physics performance studies as well as to
probe the Standard Model. The invariant mass of the _𝑍_ boson obtained when using Geant4, AtlFastII,
and AtlFast3 is shown in Figures 40(a) and 40(b), with the _𝑍_ boson reconstructed from either a pair of
muons or a pair of electrons. The visible invariant mass [72] of off-shell 2.0–2.25 TeV _𝑍_ _[★]_ / _𝛾_ _[★]_ bosons
reconstructed using the hadronic decay modes of two _𝜏_ -leptons is shown in Figure 40(c), where the width
is slightly overestimated by AtlFast3. Otherwise, no significant differences are observed between the three
distributions, which validates the simulation of prompt muons in AtlFast3.


The performance of the simulation for jet substructure is evaluated with the _𝑍_ [′] → _𝑡𝑡_ ¯ events as the benchmark
for ‘beyond the Standard Model’ signatures with boosted high- _𝑝_ T objects in the calorimeter. The leading
jet’s mass with its constituents calibrated to the EM scale is shown in Figure 41; the distribution has a
large peak near the mass of the top quark, and a small peak at the _𝑊_ boson mass. AtlFastII significantly
underestimates the mass and the width of both peaks compared to Geant4, while AtlFast3 is in better
agreement with the Geant4 distribution than AtlFastII.


48


Figure 39: The reconstructed diphoton invariant mass distribution from a selection targeting events with Higgs boson
decays into two photons. Events are selected by requiring two photons with _𝑝_ T _>_ 0 _._ 35 _𝑚_ _𝛾𝛾_ and _𝑝_ T _>_ 0 _._ 25 _𝑚_ _𝛾𝛾_, and
with | _𝜂_ | _<_ 1 _._ 37 or 1 _._ 52 _<_ | _𝜂_ | _<_ 2 _._ 47 . The statistical uncertainties are shown but may be smaller than the size of the
markers.


49


|12000 GeV<br>ATLAS Simulation<br>10000 s = 13 TeV, Z→µµ G4 1.5<br>Mean 90.24 ± 0.02 /<br>RMS 4.70 ± 0.01 events<br>8000 AF2<br>Mean 90.25 ± 0.02<br>RMS 4.70 ± 0.01 of<br>6000 AF3 Number<br>Mean 90.25 ± 0.02<br>RMS 4.66 ± 0.01<br>4000<br>2000|3000 ATLAS Simulation GeV<br>Z→ee, s=13 TeV AF2<br>2500 Mean 89.17 ± 0.03 0.6<br>G4 RMS 5.25 ± 0.02<br>/<br>Mean 89.73 ± 0.02 AF3 events<br>2000 RMS 5.06 ± 0.02 Mean 89.53 ± 0.03<br>RMS 5.19 ± 0.02<br>1500 of<br>Number<br>1000<br>500<br>0<br>G4<br>1.4<br>1.2 F /|
|---|---|
|~~80~~<br>~~85~~<br>~~90~~<br>~~95~~<br>~~100~~<br>~~105~~<br> [GeV]<br>µ<br>µ<br>m<br>0.8<br>0.9<br>~~1~~<br>1.1<br>1.2<br>AF/G4|~~80~~<br>~~85~~<br>~~90~~<br>~~95~~<br>~~100~~<br>~~105~~<br> [GeV]<br>µ<br>µ<br>m<br>0.8<br>0.9<br>~~1~~<br>1.1<br>1.2<br>AF/G4|
|~~80~~<br>~~85~~<br>~~90~~<br>~~95~~<br>~~100~~<br>~~105~~<br> [GeV]<br>µ<br>µ<br>m<br>0.8<br>0.9<br>~~1~~<br>1.1<br>1.2<br>AF/G4|[GeV]<br>ee<br>m<br>7~~5~~<br>~~80~~<br>~~85~~<br>~~90~~<br>~~95~~<br>~~100~~<br>~~10~~5<br>A<br>0.8<br>1<br>|


(b)



(a)









(c)



Figure 40: Invariant mass distribution from a selection targeting events with a _𝑍_ boson decaying into (a) two muons
or (b) two electrons with _𝑝_ T _>_ 25 GeV and | _𝜂_ | _<_ 1 _._ 37 or 1 _._ 52 _<_ | _𝜂_ | _<_ 2 _._ 47, and (c) the visible part of the invariant
mass of two hadronically decaying _𝜏_ -leptons in Drell–Yan _𝑍_ _[★]_ / _𝛾_ _[★]_ → _𝜏𝜏_ events filtered for an off-shell mass of
2.0–2.25 TeV. The statistical uncertainties are shown but may be smaller than the size of the markers.


50


|0.25<br>Normalized<br>ATLAS Simulation<br>s=13 TeV, W'(13 TeV)→WZ→4q<br>0.2 Jet p>20 GeV, EMPFlow R=0.4 jets<br>T G4<br>0.15 AF2<br>AF3<br>Unit<br>0.1|0.07<br>Normalized<br>ATLAS Simulation<br>0.06 s=13 TeV, Z'(4 TeV)→tt<br>G4 Jet p>200 GeV, UFO R=1.0 jets<br>0.05 T<br>AF2<br>0.04 AF3<br>0.03 Unit<br>0.02|
|---|---|
|0.05<br>~~0~~<br>~~10~~<br>~~20~~<br>~~30~~<br>~~40~~<br>~~50~~<br>~~60~~<br>~~70~~<br>Leading-Jet Number of Constituents<br>0.6<br>0.8<br>1<br>1.2<br>1.4<br>AF/G4|0.01<br><br>~~0~~<br>~~50~~<br>~~100~~<br>~~150~~<br>~~200~~<br>~~250~~<br>~~300~~<br>~~350~~<br>~~40~~0<br>Leading-Jet Mass [GeV]<br>0.8<br>1<br>1.2<br>AF/G4|
|||


(a)



(b)



Figure 41: Distribution of the (left) number of constituents in the leading _𝑅_ = 0 _._ 4 EMPFlow jets in the _𝑊_ [′] sample
and (right) the mass of trimmed _𝑅_ = 1 _._ 0 UFO jets in the _𝑍_ [′] sample in Geant4 (black triangles), AtlFastII (blue stars),
and AtlFast3 (red diamonds). The statistical uncertainties are shown but may be smaller than the size of the markers.


51


**8.3 Computing performance with AtlFast3**


The time required to simulate a particle in Geant4 increases with energy due to increasing shower depth
and complexity, whereas in AtlFastII and AtlFast3 the time is independent of the particle energy because it
requires a single lookup in the parameterization file. To illustrate this, the average CPU time, calculated
with a 4-core Intel i7-3770 CPU at 3.40 GHz, required to simulate a single photon produced on the
calorimeter surface at 0 _._ 20 _<_ | _𝜂_ | _<_ 0 _._ 25 is shown in Figure 42 as a function of energy. For an 8 GeV
photon produced on the calorimeter surface, AtlFast3 is approximately 20 times faster than Geant4, while
for a 256 GeV photon, AtlFast3 is approximately 600 times faster.


For the full detector simulation, the computing performance of AtlFast3 is compared with that of Geant4
by simulating the same 1000 _𝑡𝑡_ ¯ events; this is a complex process ideal for a variety of benchmarking needs
and is used extensively by the ATLAS experiment for this purpose. Each simulation algorithm is executed
on a 8-core Intel Xeon E5 CPU at 3.20 GHz. On average, Geant4 requires 167 seconds to simulate a single
event, while AtlFast3 only requires 32 seconds, thereby obtaining a speed-up of the simulation by a factor
of five. If the calorimeter simulation alone is considered, AtlFast3 is _𝑂_ ( 500 ) times faster than Geant4.
This means that the simulation time is dominated by the simulation of the inner detector performed by
Geant4. Therefore, further gains in the simulation speed of physics samples would require the use of
fast simulation techniques in the tracking detector. Due to the size of the parameterization file, AtlFast3
requires 7 GB of proportional set size (PSS) memory, while the full simulation requires 2.7 GB in total
when using eight separate cores. The parameterization requires 5 GB of PSS memory and this is shared by
the cores and is within the PSS memory budget available. The PSS memory required by AtlFast3 can be
reduced in the future through the use of compression algorithms.









Figure 42: Comparison of the CPU performance of AtlFast3 with Geant4 and AtlFastII. The average CPU time
to simulate an event is estimated using 10 000 single photons at 0 _._ 20 _<_ | _𝜂_ | _<_ 0 _._ 25 for three different energies:
8 GeV, 65 GeV, and 256 GeV . These photons are generated on the calorimeter surface and provide a comparison for
calorimeter-only simulation time.


52


#### **9 Conclusion**

An updated version of the fast simulation for the ATLAS experiment, AtlFast3, is introduced in this paper.
AtlFast3 significantly improves the modelling of reconstructed objects for physics analyses beyond that
obtained by AtlFastII. In most cases, AtlFast3 and Geant4 agree to within a few percent. Key improvements
include the modelling of the response in the forward calorimeters and of shower substructure within jets.
Moreover, AtlFast3 requires only 20% as much CPU as Geant4 to simulate an event. The version of
AtlFast3 described in this paper is currently being used by ATLAS to simulate 7 billion events for physics
analyses of the Run 2 data. Further updates and improvements to the modelling are anticipated for Run 3
and beyond.

#### **Acknowledgements**


We thank CERN for the very successful operation of the LHC, as well as the support staff from our
institutions without whom ATLAS could not be operated efficiently.


We acknowledge the support of ANPCyT, Argentina; YerPhI, Armenia; ARC, Australia; BMWFW and
FWF, Austria; ANAS, Azerbaijan; SSTC, Belarus; CNPq and FAPESP, Brazil; NSERC, NRC and CFI,
Canada; CERN; ANID, Chile; CAS, MOST and NSFC, China; Minciencias, Colombia; MSMT CR, MPO
CR and VSC CR, Czech Republic; DNRF and DNSRC, Denmark; IN2P3-CNRS and CEA-DRF/IRFU,
France; SRNSFG, Georgia; BMBF, HGF and MPG, Germany; GSRI, Greece; RGC and Hong Kong SAR,
China; ISF and Benoziyo Center, Israel; INFN, Italy; MEXT and JSPS, Japan; CNRST, Morocco; NWO,
Netherlands; RCN, Norway; MNiSW and NCN, Poland; FCT, Portugal; MNE/IFA, Romania; JINR; MES
of Russia and NRC KI, Russian Federation; MESTD, Serbia; MSSR, Slovakia; ARRS and MIZS, Slovenia; [ˇ]
DSI/NRF, South Africa; MICINN, Spain; SRC and Wallenberg Foundation, Sweden; SERI, SNSF and
Cantons of Bern and Geneva, Switzerland; MOST, Taiwan; TAEK, Turkey; STFC, United Kingdom; DOE
and NSF, United States of America. In addition, individual groups and members have received support
from BCKDF, CANARIE, Compute Canada and CRC, Canada; COST, ERC, ERDF, Horizon 2020 and
Marie Sk lodowska-Curie Actions, European Union; Investissements d’Avenir Labex, Investissements
d’Avenir Idex and ANR, France; DFG and AvH Foundation, Germany; Herakleitos, Thales and Aristeia
programmes co-financed by EU-ESF and the Greek NSRF, Greece; BSF-NSF and GIF, Israel; Norwegian
Financial Mechanism 2014-2021, Norway; La Caixa Banking Foundation, CERCA Programme Generalitat
de Catalunya and PROMETEO and GenT Programmes Generalitat Valenciana, Spain; Goran Gustafssons ¨
Stiftelse, Sweden; The Royal Society and Leverhulme Trust, United Kingdom.


The crucial computing support from all WLCG partners is acknowledged gratefully, in particular from
CERN, the ATLAS Tier-1 facilities at TRIUMF (Canada), NDGF (Denmark, Norway, Sweden), CC-IN2P3
(France), KIT/GridKA (Germany), INFN-CNAF (Italy), NL-T1 (Netherlands), PIC (Spain), ASGC
(Taiwan), RAL (UK) and BNL (USA), the Tier-2 facilities worldwide and large non-WLCG resource
providers. Major contributors of computing resources are listed in Ref. [73].


53


#### **References**


[1] ATLAS Collaboration, _The ATLAS Experiment at the CERN Large Hadron Collider_,
JINST **3** [(2008) S08003.](https://doi.org/10.1088/1748-0221/3/08/S08003)


[2] L. Evans and P. Bryant, _LHC Machine_, JINST **3** [(2008) S08001.](https://doi.org/10.1088/1748-0221/3/08/S08001)


[3] ATLAS Collaboration, _ATLAS HL-LHC Computing Conceptual Design Report_, tech. rep.,
CERN, 2020, url: `[https://cds.cern.ch/record/2729668](https://cds.cern.ch/record/2729668)` .


[4] H. S. Foundation et al.,
_HEP Software Foundation Community White Paper Working Group - Detector Simulation_, 2018,
arXiv: `[1803.04165 [physics.comp-ph]](https://arxiv.org/abs/1803.04165)` .


[5] GEANT4 Collaboration, S. Agostinelli, et al., _Geant4 – a simulation toolkit_,
[Nucl. Instrum. Meth. A](https://doi.org/10.1016/S0168-9002(03)01368-8) **506** (2003) 250.


[6] ATLAS Collaboration, _The ATLAS Simulation Infrastructure_, Eur. Phys. J. C **[70](https://doi.org/10.1140/epjc/s10052-010-1429-9)** (2010) 823,
arXiv: `[1005.4568 [physics.ins-det]](https://arxiv.org/abs/1005.4568)` .


[7] ATLAS Collaboration,
_The simulation principle and performance of the ATLAS fast calorimeter simulation FastCaloSim_,
ATL-PHYS-PUB-2010-013, 2010, url: `[https://cds.cern.ch/record/1300517](https://cds.cern.ch/record/1300517)` .


[8] I. J. Goodfellow et al., _Generative Adversarial Networks_, 2014, arXiv: `[1406.2661 [stat.ML]](https://arxiv.org/abs/1406.2661)` .


[9] ATLAS Collaboration,
_Fast simulation of the ATLAS calorimeter system with Generative Adversarial Networks_,
ATL-SOFT-PUB-2020-006, 2020, url: `[https://cds.cern.ch/record/2746032](https://cds.cern.ch/record/2746032)` .


[10] ATLAS Collaboration, _ATLAS Insertable B-Layer: Technical Design Report_,
ATLAS-TDR-19; CERN-LHCC-2010-013, 2010,
url: `[https://cds.cern.ch/record/1291633](https://cds.cern.ch/record/1291633)`, Addendum: ATLAS-TDR-19-ADD-1;
CERN-LHCC-2012-009, 2012, url: `[https://cds.cern.ch/record/1451888](https://cds.cern.ch/record/1451888)` .


[11] ATLAS Collaboration, _The ATLAS Collaboration Software and Firmware_,
ATL-SOFT-PUB-2021-001, 2021, url: `[https://cds.cern.ch/record/2767187](https://cds.cern.ch/record/2767187)` .


[12] _ATLAS calorimeter performance: Technical Design Report_, Technical Design Report ATLAS,
Geneva: CERN, 1996, url: `[https://cds.cern.ch/record/331059](https://cds.cern.ch/record/331059)` .


[13] J. Allison et al., _Recent developments in Geant4_ [, Nucl. Instrum. Meth. A](https://doi.org/10.1016/j.nima.2016.06.125) **835** (2016) 186.


[14] B. Andersson, G. Gustafson, and B. Nilsson-Almqvist, _A model for low-_ _𝑝_ _𝑇_ _hadronic reactions, with_
_generalizations to hadron-nucleus and nucleus-nucleus collisions_, Nucl. Phys. B **[281](https://doi.org/10.1016/0550-3213(87)90257-4)** (1987) 289.


[15] B. Andersson, A. Tai, and B.-H. Sa,
_Final state interactions in the (nuclear) FRITIOF string interaction scenario_,

Z. Phys. C **[70](https://doi.org/10.1007/s002880050127)** (1996) 499.


[16] B. Ganhuyag and V. Uzhinsky, _Modified FRITIOF code: negative charged particle production in_
_high energy nucleus–nucleus interactions_,
[Czech. J. Phys.](https://doi.org/10.1023/A:1021296114786) **47** (1997) 913, ed. by A. Kugler, J. Dolejsi, and I. Hrivnacova.


[17] T. Jeˇzo, J. M. Lindert, N. Moretti, and S. Pozzorini,
_New NLOPS predictions for 𝑡𝑡_ ¯ + _𝑏-jet production at the LHC_, Eur. Phys. J. C **[78](https://doi.org/10.1140/epjc/s10052-018-5956-0)** (2018) 502,
arXiv: `[1802.00426 [hep-ph]](https://arxiv.org/abs/1802.00426)` .


54


[18] T. Sj¨ostrand, S. Mrenna, and P. Z. Skands, _PYTHIA 6.4 physics and manual_, JHEP **05** [(2006) 026,](https://doi.org/10.1088/1126-6708/2006/05/026)
arXiv: `[hep-ph/0603175](https://arxiv.org/abs/hep-ph/0603175)` .


[19] J. Gao et al., _CT10 next-to-next-to-leading order global analysis of QCD_,
Phys. Rev. D **[89](https://doi.org/10.1103/PhysRevD.89.033009)** (2014) 033009, arXiv: `[1302.6246 [hep-ph]](https://arxiv.org/abs/1302.6246)` .


[20] P. Z. Skands, _Tuning Monte Carlo generators: The Perugia tunes_, Phys. Rev. D **[82](https://doi.org/10.1103/PhysRevD.82.074018)** (2010) 074018,
arXiv: `[1005.3457 [hep-ph]](https://arxiv.org/abs/1005.3457)` .


[21] S. Frixione and B. R. Webber, _Matching NLO QCD computations and parton shower simulations_,
JHEP **06** [(2002) 029, arXiv:](https://doi.org/10.1088/1126-6708/2002/06/029) `[hep-ph/0204244](https://arxiv.org/abs/hep-ph/0204244)` .


[22] J. Pumplin et al.,
_New Generation of Parton Distributions with Uncertainties from Global QCD Analysis_,
JHEP **07** [(2002) 012, arXiv:](https://doi.org/10.1088/1126-6708/2002/07/012) `[hep-ph/0201195](https://arxiv.org/abs/hep-ph/0201195)` .


[23] T. Sj¨ostrand, S. Mrenna, and P. Skands, _A brief introduction to PYTHIA 8.1_,
[Comput. Phys. Commun.](https://doi.org/10.1016/j.cpc.2008.01.036) **178** (2008) 852, arXiv: `[0710.3820 [hep-ph]](https://arxiv.org/abs/0710.3820)` .


[24] ATLAS Collaboration, _Measurement of the 𝑍_ / _𝛾_ [∗] _boson transverse momentum distribution in 𝑝𝑝_
_collisions at_ ~~[√]~~ ~~_𝑠_~~ = 7 _TeV with the ATLAS detector_, JHEP **09** [(2014) 145,](https://doi.org/10.1007/JHEP09(2014)145)
arXiv: `[1406.3660 [hep-ex]](https://arxiv.org/abs/1406.3660)` .


[25] S. Carrazza, S. Forte, and J. Rojo, _Parton Distributions and Event Generators_, 2013,
arXiv: `[1311.5887 [hep-ph]](https://arxiv.org/abs/1311.5887)` .


[26] D. J. Lange, _The EvtGen particle decay simulation package_,
[Nucl. Instrum. Meth. A](https://doi.org/10.1016/S0168-9002(01)00089-4) **462** (2001) 152.


[27] R. D. Ball et al., _Parton distributions with LHC data_, Nucl. Phys. B **[867](https://doi.org/10.1016/j.nuclphysb.2012.10.003)** (2013) 244,
arXiv: `[1207.1303 [hep-ph]](https://arxiv.org/abs/1207.1303)` .


[28] ATLAS Collaboration, _ATLAS Pythia 8 tunes to_ 7 _TeV data_, ATL-PHYS-PUB-2014-021, 2014,
url: `[https://cds.cern.ch/record/1966419](https://cds.cern.ch/record/1966419)` .


[29] P. Nason, _A new method for combining NLO QCD with shower Monte Carlo algorithms_,
JHEP **11** [(2004) 040, arXiv:](https://doi.org/10.1088/1126-6708/2004/11/040) `[hep-ph/0409146](https://arxiv.org/abs/hep-ph/0409146)` .


[30] S. Frixione, P. Nason, and C. Oleari,
_Matching NLO QCD computations with parton shower simulations: the POWHEG method_,
JHEP **11** [(2007) 070, arXiv:](https://doi.org/10.1088/1126-6708/2007/11/070) `[0709.2092 [hep-ph]](https://arxiv.org/abs/0709.2092)` .


[31] S. Alioli, P. Nason, C. Oleari, and E. Re, _A general framework for implementing NLO calculations_
_in shower Monte Carlo programs: the POWHEG BOX_, JHEP **06** [(2010) 043,](https://doi.org/10.1007/JHEP06(2010)043)
arXiv: `[1002.2581 [hep-ph]](https://arxiv.org/abs/1002.2581)` .


[32] K. Hamilton, P. Nason, E. Re, and G. Zanderighi, _NNLOPS simulation of Higgs boson production_,
JHEP **10** [(2013) 222, arXiv:](https://doi.org/10.1007/JHEP10(2013)222) `[1309.0017 [hep-ph]](https://arxiv.org/abs/1309.0017)` .


[33] K. Hamilton, P. Nason, and G. Zanderighi,
_Finite quark-mass effects in the NNLOPS POWHEG+MiNLO Higgs generator_, JHEP **05** [(2015) 140,](https://doi.org/10.1007/JHEP05(2015)140)
arXiv: `[1501.04637 [hep-ph]](https://arxiv.org/abs/1501.04637)` .


[34] K. Hamilton, P. Nason, and G. Zanderighi, _MINLO: Multi-Scale Improved NLO_,
JHEP **10** [(2012) 155, arXiv:](https://doi.org/10.1007/JHEP10(2012)155) `[1206.3572 [hep-ph]](https://arxiv.org/abs/1206.3572)` .


[35] J. M. Campbell et al., _NLO Higgs Boson Production Plus One and Two Jets Using the POWHEG_
_BOX, MadGraph4 and MCFM_, JHEP **07** [(2012) 092, arXiv:](https://doi.org/10.1007/JHEP07(2012)092) `[1202.5475 [hep-ph]](https://arxiv.org/abs/1202.5475)` .


55


[36] K. Hamilton, P. Nason, C. Oleari, and G. Zanderighi, _Merging H/W/Z + 0 and 1 jet at NLO with no_
_merging scale: a path to parton shower + NNLO matching_, JHEP **05** [(2013) 082,](https://doi.org/10.1007/JHEP05(2013)082)
arXiv: `[1212.4504 [hep-ph]](https://arxiv.org/abs/1212.4504)` .


[37] S. Catani and M. Grazzini, _An NNLO subtraction formalism in hadron collisions and its application_
_to Higgs boson production at the LHC_, Phys. Rev. Lett. **[98](https://doi.org/10.1103/PhysRevLett.98.222002)** (2007) 222002,
arXiv: `[hep-ph/0703012 [hep-ph]](https://arxiv.org/abs/hep-ph/0703012)` .


[38] J. Butterworth et al., _PDF4LHC recommendations for LHC Run II_, J. Phys. G **43** [(2016) 023001,](https://doi.org/10.1088/0954-3899/43/2/023001)
arXiv: `[1510.03865 [hep-ph]](https://arxiv.org/abs/1510.03865)` .


[39] ATLAS Collaboration, _Emulating the impact of additional proton–proton interactions in the ATLAS_
_simulation by pre-sampling sets of inelastic Monte Carlo events_, submitted to CSBS (2021),
arXiv: `[2102.09495 [hep-ex]](https://arxiv.org/abs/2102.09495)` .


[40] ATLAS Collaboration, _The Pythia 8 A3 tune description of ATLAS minimum bias and inelastic_
_measurements incorporating the Donnachie–Landshoff diffractive model_,
ATL-PHYS-PUB-2016-017, 2016, url: `[https://cds.cern.ch/record/2206965](https://cds.cern.ch/record/2206965)` .


[41] ATLAS Collaboration, _Reconstruction of primary vertices at the ATLAS experiment in Run 1_
_proton–proton collisions at the LHC_, Eur. Phys. J. C **[77](https://doi.org/10.1140/epjc/s10052-017-4887-5)** (2017) 332,
arXiv: `[1611.10235 [hep-ex]](https://arxiv.org/abs/1611.10235)` .


[42] ATLAS Collaboration, _Electron and photon performance measurements with the ATLAS detector_
_using the 2015–2017 LHC proton–proton collision data_, JINST **14** [(2019) P12006,](https://doi.org/10.1088/1748-0221/14/12/P12006)
arXiv: `[1908.00005 [hep-ex]](https://arxiv.org/abs/1908.00005)` .


[43] ATLAS Collaboration, _Muon reconstruction and identification efficiency in ATLAS using the full_
_Run 2 𝑝𝑝_ _collision data set at_ ~~[√]~~ ~~_𝑠_~~ = 13 _TeV_, Eur. Phys. J. C **[81](https://doi.org/10.1140/epjc/s10052-021-09233-2)** (2021) 578,
arXiv: `[2012.00578 [hep-ex]](https://arxiv.org/abs/2012.00578)` .


[44] M. Cacciari, G. P. Salam, and G. Soyez, _The anti-𝑘_ _𝑡_ _jet clustering algorithm_, JHEP **04** [(2008) 063,](https://doi.org/10.1088/1126-6708/2008/04/063)
arXiv: `[0802.1189 [hep-ph]](https://arxiv.org/abs/0802.1189)` .


[45] ATLAS Collaboration,
_Jet reconstruction and performance using particle flow with the ATLAS Detector_,
Eur. Phys. J. C **[77](https://doi.org/10.1140/epjc/s10052-017-5031-2)** (2017) 466, arXiv: `[1703.10485 [hep-ex]](https://arxiv.org/abs/1703.10485)` .


[46] ATLAS Collaboration, _Performance of missing transverse momentum reconstruction with the_
_ATLAS detector using proton–proton collisions at_ ~~[√]~~ ~~_𝑠_~~ = 13 _TeV_, Eur. Phys. J. C **[78](https://doi.org/10.1140/epjc/s10052-018-6288-9)** (2018) 903,
arXiv: `[1802.08168 [hep-ex]](https://arxiv.org/abs/1802.08168)` .


[47] ATLAS Collaboration, _Jet energy scale and resolution measured in proton–proton collisions at_
~~√~~ ~~_𝑠_~~ = 13 _TeV with the ATLAS detector_, Eur. Phys. J. C **[81](https://doi.org/10.1140/epjc/s10052-021-09402-3)** (2021) 689,
arXiv: `[2007.02645 [hep-ex]](https://arxiv.org/abs/2007.02645)` .


[48] I. Jolliffe, “Principal Component Analysis,” _International Encyclopedia of Statistical Science_,
ed. by M. Lovric, Berlin, Heidelberg: Springer Berlin Heidelberg, 2011 1094,
isbn: 978-3-642-04898-2, url: `[https://doi.org/10.1007/978-3-642-04898-2_455](https://doi.org/10.1007/978-3-642-04898-2_455)` .


[49] ATLAS Collaboration,
_Topological cell clustering in the ATLAS calorimeters and its performance in LHC Run 1_,

Eur. Phys. J. C **[77](https://doi.org/10.1140/epjc/s10052-017-5004-5)** (2017) 490, arXiv: `[1603.02934 [hep-ex]](https://arxiv.org/abs/1603.02934)` .


[50] ATLAS Collaboration, _Deep generative models for fast shower simulation in ATLAS_,
ATL-SOFT-PUB-2018-001, 2018, url: `[https://cds.cern.ch/record/2630433](https://cds.cern.ch/record/2630433)` .


56


[51] L. de Oliveira, M. Paganini, and B. Nachman, _Learning Particle Physics by Example:_
_Location-Aware Generative Adversarial Networks for Physics Synthesis_,
[Comput. Softw. Big Sci.](https://doi.org/10.1007/s41781-017-0004-6) **1** (2017) 4, issn: 2510-2044, arXiv: `[1701.05927](https://arxiv.org/abs/1701.05927)` .


[52] M. Paganini, L. de Oliveira, and B. Nachman, _Accelerating Science with Generative Adversarial_
_Networks: An Application to 3D Particle Showers in Multilayer Calorimeters_,
Phys. Rev. Lett. **[120](https://doi.org/10.1103/PhysRevLett.120.042003)** (2018) 042003, arXiv: `[1705.02355](https://arxiv.org/abs/1705.02355)` .


[53] M. Paganini, L. de Oliveira, and B. Nachman, _CaloGAN: Simulating 3D high energy particle_
_showers in multilayer electromagnetic calorimeters with generative adversarial networks_,
Phys. Rev. D **[97](https://doi.org/10.1103/physrevd.97.014021)** (2018) 014021, arXiv: `[1712.10321](https://arxiv.org/abs/1712.10321)` .


[54] M. Erdmann, L. Geiger, J. Glombitza, and D. Schmidt, _Generating and refining particle detector_
_simulations using the Wasserstein distance in adversarial networks_, 2018,
arXiv: `[1802.03325 [astro-ph.IM]](https://arxiv.org/abs/1802.03325)` .


[55] M. Erdmann, J. Glombitza, and T. Quast, _Precise Simulation of Electromagnetic Calorimeter_
_Showers Using a Wasserstein Generative Adversarial Network_ [, Comput. Softw. Big Sci.](https://doi.org/10.1007/s41781-018-0019-7) **3** (2019) 4,
arXiv: `[1807.01954](https://arxiv.org/abs/1807.01954)` .


[56] F. Carminati et al., _Three dimensional Generative Adversarial Networks for fast simulation_,
J. Phys: Conf. Ser. **[1085](https://doi.org/10.1088/1742-6596/1085/3/032016)** (2018) 032016.


[57] M. Arjovsky, S. Chintala, and L. Bottou, _Wasserstein GAN_, 2017, arXiv: `[1701.07875 [stat.ML]](https://arxiv.org/abs/1701.07875)` .


[58] I. Gulrajani, F. Ahmed, M. Arjovsky, V. Dumoulin, and A. Courville,
_Improved Training of Wasserstein GANs_, 2017, arXiv: `[1704.00028 [cs.LG]](https://arxiv.org/abs/1704.00028)` .


[59] M. Abadi et al., _TensorFlow: Large-Scale Machine Learning on Heterogeneous Distributed Systems_,
(2016), arXiv: `[1603.04467 [cs.DC]](https://arxiv.org/abs/1603.04467)` .


[60] D. P. Kingma and J. Ba, _Adam: A Method for Stochastic Optimization_, 2014,
arXiv: `[1412.6980 [cs.LG]](https://arxiv.org/abs/1412.6980)` .


[61] _NVIDIA V100 architecture_, 2020,
url: `[https://www.nvidia.com/en-us/data-center/volta-gpu-architecture/](https://www.nvidia.com/en-us/data-center/volta-gpu-architecture/)` .


[62] _The HTCondor batch system at CERN_, 2021,
url: `[https://batchdocs.web.cern.ch/index.html](https://batchdocs.web.cern.ch/index.html)` .


[63] D. H. Guest et al., _lwtnn/lwtnn: Version 2.9_, version v2.9, 2019,
url: `[https://doi.org/10.5281/zenodo.3249317](https://doi.org/10.5281/zenodo.3249317)` .


[64] W. Lukas, _Fast Simulation for ATLAS: Atlfast-II and ISF_,
J. Phys. Conf. Ser. **[396](https://doi.org/10.1088/1742-6596/396/2/022031)** (2012) 022031, ed. by M. Ernst, D. D¨ullmann, O. Rind, and T. Wong.


[65] M. Cacciari, G. P. Salam, and G. Soyez, _FastJet user manual_, Eur. Phys. J. C **[72](https://doi.org/10.1140/epjc/s10052-012-1896-2)** (2012) 1896,
arXiv: `[1111.6097 [hep-ph]](https://arxiv.org/abs/1111.6097)` .


[66] ATLAS Collaboration, _Optimisation of large-radius jet reconstruction for the ATLAS detector in 13_
_TeV proton–proton collisions_, 2020, arXiv: `[2009.04986 [hep-ex]](https://arxiv.org/abs/2009.04986)` .


[67] J. Thaler and K. Van Tilburg, _Identifying boosted objects with N-subjettiness_, JHEP **03** [(2011) 015.](https://doi.org/10.1007/jhep03(2011)015)


[68] A. J. Larkoski, I. Moult, and D. Neill, _Power counting to better jet observables_,
JHEP **12** [(2014) 009, arXiv:](https://doi.org/10.1007/JHEP12(2014)009) `[1409.6298](https://arxiv.org/abs/1409.6298)` .


[69] ATLAS Collaboration,
_Reconstruction of hadronic decay products of tau leptons with the ATLAS experiment_,
Eur. Phys. J. C **[76](https://doi.org/10.1140/epjc/s10052-016-4110-0)** (2016) 295, arXiv: `[1512.05955 [hep-ex]](https://arxiv.org/abs/1512.05955)` .


57


[70] ATLAS Collaboration, _Reconstruction, Energy Calibration, and Identification of Hadronically_
_Decaying Tau Leptons in the ATLAS Experiment for Run-2 of the LHC_, ATL-PHYS-PUB-2015-045,
2015, url: `[https://cds.cern.ch/record/2064383](https://cds.cern.ch/record/2064383)` .


[71] ATLAS Collaboration, _Measurement of the tau lepton reconstruction and identification_
_performance in the ATLAS experiment using 𝑝𝑝_ _collisions at_ ~~[√]~~ ~~_𝑠_~~ = 13 _TeV_,
ATLAS-CONF-2017-029, 2017, url: `[https://cds.cern.ch/record/2261772](https://cds.cern.ch/record/2261772)` .


[72] ATLAS Collaboration,
_Reconstruction of hadronic decay products of tau leptons with the ATLAS experiment_,
Eur. Phys. J. C **[76](https://doi.org/10.1140/epjc/s10052-016-4110-0)** (2016) 295, arXiv: `[1512.05955 [hep-ex]](https://arxiv.org/abs/1512.05955)` .


[73] ATLAS Collaboration, _ATLAS Computing Acknowledgements_, ATL-SOFT-PUB-2021-003,
url: `[https://cds.cern.ch/record/2776662](https://cds.cern.ch/record/2776662)` .


58


#### The ATLAS Collaboration

[G. Aad](https://orcid.org/0000-0002-6665-4934) [98] [, B. Abbott](https://orcid.org/0000-0002-5888-2734) [124] [, D.C. Abbott](https://orcid.org/0000-0002-7248-3203) [99] [, A. Abed Abud](https://orcid.org/0000-0002-2788-3822) [34] [, K. Abeling](https://orcid.org/0000-0002-1002-1652) [51] [, D.K. Abhayasinghe](https://orcid.org/0000-0002-2987-4006) [91],
[S.H. Abidi](https://orcid.org/0000-0002-8496-9294) [27] [, A. Aboulhorma](https://orcid.org/0000-0002-9987-2292) [33e] [, H. Abramowicz](https://orcid.org/0000-0001-5329-6640) [157] [, H. Abreu](https://orcid.org/0000-0002-1599-2896) [156] [, Y. Abulaiti](https://orcid.org/0000-0003-0403-3697) [5],
[A.C. Abusleme Hoffman](https://orcid.org/0000-0003-0762-7204) [142a] [, B.S. Acharya](https://orcid.org/0000-0002-8588-9157) [64a,64b,o] [, B. Achkar](https://orcid.org/0000-0002-0288-2567) [51] [, L. Adam](https://orcid.org/0000-0001-6005-2812) [96] [, C. Adam Bourdarios](https://orcid.org/0000-0002-2634-4958) [4],
[L. Adamczyk](https://orcid.org/0000-0002-5859-2075) [81a] [, L. Adamek](https://orcid.org/0000-0003-1562-3502) [162] [, S.V. Addepalli](https://orcid.org/0000-0002-2919-6663) [24] [, J. Adelman](https://orcid.org/0000-0002-1041-3496) [116] [, A. Adiguzel](https://orcid.org/0000-0001-6644-0517) [11c,ac] [, S. Adorni](https://orcid.org/0000-0003-3620-1149) [52],
[T. Adye](https://orcid.org/0000-0003-0627-5059) [139] [, A.A. Affolder](https://orcid.org/0000-0002-9058-7217) [141] [, Y. Afik](https://orcid.org/0000-0001-8102-356X) [34] [, C. Agapopoulou](https://orcid.org/0000-0002-2368-0147) [62] [, M.N. Agaras](https://orcid.org/0000-0002-4355-5589) [12] [, J. Agarwala](https://orcid.org/0000-0002-4754-7455) [68a,68b],
[A. Aggarwal](https://orcid.org/0000-0002-1922-2039) [114] [, C. Agheorghiesei](https://orcid.org/0000-0003-3695-1847) [25c] [, J.A. Aguilar-Saavedra](https://orcid.org/0000-0002-5475-8920) [135f,135a,ab] [, A. Ahmad](https://orcid.org/0000-0001-8638-0582) [34] [, F. Ahmadov](https://orcid.org/0000-0003-3644-540X) [77],
[W.S. Ahmed](https://orcid.org/0000-0003-0128-3279) [100] [, X. Ai](https://orcid.org/0000-0003-3856-2415) [44] [, G. Aielli](https://orcid.org/0000-0002-0573-8114) [71a,71b] [, I. Aizenberg](https://orcid.org/0000-0003-2150-1624) [175] [, S. Akatsuka](https://orcid.org/0000-0002-1681-6405) [83] [, M. Akbiyik](https://orcid.org/0000-0002-7342-3130) [96], T.P.A. [˚] [Akesson](https://orcid.org/0000-0003-4141-5408) [94],
[A.V. Akimov](https://orcid.org/0000-0002-2846-2958) [107] [, K. Al Khoury](https://orcid.org/0000-0002-0547-8199) [37] [, G.L. Alberghi](https://orcid.org/0000-0003-2388-987X) [21b] [, J. Albert](https://orcid.org/0000-0003-0253-2505) [171] [, P. Albicocco](https://orcid.org/0000-0001-6430-1038) [49] [, M.J. Alconada Verzini](https://orcid.org/0000-0003-2212-7830) [86],
[S. Alderweireldt](https://orcid.org/0000-0002-8224-7036) [48] [, M. Aleksa](https://orcid.org/0000-0002-1936-9217) [34] [, I.N. Aleksandrov](https://orcid.org/0000-0001-7381-6762) [77] [, C. Alexa](https://orcid.org/0000-0003-0922-7669) [25b] [, T. Alexopoulos](https://orcid.org/0000-0002-8977-279X) [9] [, A. Alfonsi](https://orcid.org/0000-0001-7406-4531) [115],
[F. Alfonsi](https://orcid.org/0000-0002-0966-0211) [21b] [, M. Alhroob](https://orcid.org/0000-0001-7569-7111) [124] [, B. Ali](https://orcid.org/0000-0001-8653-5556) [137] [, S. Ali](https://orcid.org/0000-0001-5216-3133) [154] [, M. Aliev](https://orcid.org/0000-0002-9012-3746) [161] [, G. Alimonti](https://orcid.org/0000-0002-7128-9046) [66a] [, C. Allaire](https://orcid.org/0000-0003-4745-538X) [34],
[B.M.M. Allbrooke](https://orcid.org/0000-0002-5738-2471) [152] [, P.P. Allport](https://orcid.org/0000-0001-7303-2570) [19] [, A. Aloisio](https://orcid.org/0000-0002-3883-6693) [67a,67b] [, F. Alonso](https://orcid.org/0000-0001-9431-8156) [86] [, C. Alpigiani](https://orcid.org/0000-0002-7641-5814) [144],
E. Alunno Camelia [71a,71b] [, M. Alvarez Estevez](https://orcid.org/0000-0002-8181-6532) [95] [, M.G. Alviggi](https://orcid.org/0000-0003-0026-982X) [67a,67b] [, Y. Amaral Coutinho](https://orcid.org/0000-0002-1798-7230) [78b],
[A. Ambler](https://orcid.org/0000-0003-2184-3480) [100] [, L. Ambroz](https://orcid.org/0000-0002-0987-6637) [130], C. Amelung [34] [, D. Amidei](https://orcid.org/0000-0002-6814-0355) [102] [, S.P. Amor Dos Santos](https://orcid.org/0000-0001-7566-6067) [135a] [, S. Amoroso](https://orcid.org/0000-0001-5450-0447) [44],
[K.R. Amos](https://orcid.org/0000-0003-1757-5620) [169], C.S. Amrouche [52] [, V. Ananiev](https://orcid.org/0000-0003-3649-7621) [129] [, C. Anastopoulos](https://orcid.org/0000-0003-1587-5830) [145] [, N. Andari](https://orcid.org/0000-0002-4935-4753) [140] [, T. Andeen](https://orcid.org/0000-0002-4413-871X) [10],
[J.K. Anders](https://orcid.org/0000-0002-1846-0262) [18] [, S.Y. Andrean](https://orcid.org/0000-0002-9766-2670) [43a,43b] [, A. Andreazza](https://orcid.org/0000-0001-5161-5759) [66a,66b] [, S. Angelidakis](https://orcid.org/0000-0002-8274-6118) [8] [, A. Angerami](https://orcid.org/0000-0001-7834-8750) [37],
[A.V. Anisenkov](https://orcid.org/0000-0002-7201-5936) [117b,117a] [, A. Annovi](https://orcid.org/0000-0002-4649-4398) [69a] [, C. Antel](https://orcid.org/0000-0001-9683-0890) [52] [, M.T. Anthony](https://orcid.org/0000-0002-5270-0143) [145] [, E. Antipov](https://orcid.org/0000-0002-6678-7665) [125] [, M. Antonelli](https://orcid.org/0000-0002-2293-5726) [49],
[D.J.A. Antrim](https://orcid.org/0000-0001-8084-7786) [16] [, F. Anulli](https://orcid.org/0000-0003-2734-130X) [70a] [, M. Aoki](https://orcid.org/0000-0001-7498-0097) [79] [, J.A. Aparisi Pozo](https://orcid.org/0000-0001-7401-4331) [169] [, M.A. Aparo](https://orcid.org/0000-0003-4675-7810) [152] [, L. Aperio Bella](https://orcid.org/0000-0003-3942-1702) [44],
[N. Aranzabal](https://orcid.org/0000-0001-9013-2274) [34] [, V. Araujo Ferraz](https://orcid.org/0000-0003-1177-7563) [78a] [, C. Arcangeletti](https://orcid.org/0000-0001-8648-2896) [49] [, A.T.H. Arce](https://orcid.org/0000-0002-7255-0832) [47] [, E. Arena](https://orcid.org/0000-0001-5970-8677) [88] [, J-F. Arguin](https://orcid.org/0000-0003-0229-3858) [106],
[S. Argyropoulos](https://orcid.org/0000-0001-7748-1429) [50] [, J.-H. Arling](https://orcid.org/0000-0002-1577-5090) [44] [, A.J. Armbruster](https://orcid.org/0000-0002-9007-530X) [34] [, A. Armstrong](https://orcid.org/0000-0001-8505-4232) [166] [, O. Arnaez](https://orcid.org/0000-0002-6096-0893) [162] [, H. Arnold](https://orcid.org/0000-0003-3578-2228) [34],
Z.P. Arrubarrena Tame [110] [, G. Artoni](https://orcid.org/0000-0002-3477-4499) [130] [, H. Asada](https://orcid.org/0000-0003-1420-4955) [112] [, K. Asai](https://orcid.org/0000-0002-3670-6908) [122] [, S. Asai](https://orcid.org/0000-0001-5279-2298) [159] [, N.A. Asbah](https://orcid.org/0000-0001-8381-2255) [57],
[E.M. Asimakopoulou](https://orcid.org/0000-0003-2127-373X) [167] [, L. Asquith](https://orcid.org/0000-0001-8035-7162) [152] [, J. Assahsah](https://orcid.org/0000-0002-3207-9783) [33d], K. Assamagan [27] [, R. Astalos](https://orcid.org/0000-0001-5095-605X) [26a] [, R.J. Atkin](https://orcid.org/0000-0002-1972-1006) [31a],
M. Atkinson [168] [, N.B. Atlay](https://orcid.org/0000-0003-1094-4825) [17], H. Atmani [58b] [, P.A. Atmasiddha](https://orcid.org/0000-0002-7639-9703) [102] [, K. Augsten](https://orcid.org/0000-0001-8324-0576) [137] [, S. Auricchio](https://orcid.org/0000-0001-7599-7712) [67a,67b],
[V.A. Austrup](https://orcid.org/0000-0001-6918-9065) [177] [, G. Avner](https://orcid.org/0000-0003-1616-3587) [156] [, G. Avolio](https://orcid.org/0000-0003-2664-3437) [34] [, M.K. Ayoub](https://orcid.org/0000-0001-5265-2674) [13c] [, G. Azuelos](https://orcid.org/0000-0003-4241-022X) [106,ai] [, D. Babal](https://orcid.org/0000-0001-7657-6004) [26a] [, H. Bachacou](https://orcid.org/0000-0002-2256-4515) [140],
[K. Bachas](https://orcid.org/0000-0002-9047-6517) [158] [, A. Bachiu](https://orcid.org/0000-0001-8599-024X) [32] [, F. Backman](https://orcid.org/0000-0001-7489-9184) [43a,43b] [, A. Badea](https://orcid.org/0000-0001-5199-9588) [57] [, P. Bagnaia](https://orcid.org/0000-0003-4578-2651) [70a,70b], H. Bahrasemani [148],
[A.J. Bailey](https://orcid.org/0000-0002-3301-2986) [169] [, V.R. Bailey](https://orcid.org/0000-0001-8291-5711) [168] [, J.T. Baines](https://orcid.org/0000-0003-0770-2702) [139] [, C. Bakalis](https://orcid.org/0000-0002-9931-7379) [9] [, O.K. Baker](https://orcid.org/0000-0003-1346-5774) [178] [, P.J. Bakker](https://orcid.org/0000-0002-3479-1125) [115] [, E. Bakos](https://orcid.org/0000-0002-1110-4433) [14],
[D. Bakshi Gupta](https://orcid.org/0000-0002-6580-008X) [7] [, S. Balaji](https://orcid.org/0000-0002-5364-2109) [153] [, R. Balasubramanian](https://orcid.org/0000-0001-5840-1788) [115] [, E.M. Baldin](https://orcid.org/0000-0002-9854-975X) [117b,117a] [, P. Balek](https://orcid.org/0000-0002-0942-1966) [138],
[E. Ballabene](https://orcid.org/0000-0001-9700-2587) [66a,66b] [, F. Balli](https://orcid.org/0000-0003-0844-4207) [140] [, L.M. Baltes](https://orcid.org/0000-0001-7041-7096) [59a] [, W.K. Balunas](https://orcid.org/0000-0002-7048-4915) [130] [, J. Balz](https://orcid.org/0000-0003-2866-9446) [96] [, E. Banas](https://orcid.org/0000-0001-5325-6040) [82],
[M. Bandieramonte](https://orcid.org/0000-0003-2014-9489) [134] [, A. Bandyopadhyay](https://orcid.org/0000-0002-5256-839X) [22] [, S. Bansal](https://orcid.org/0000-0002-8754-1074) [22] [, L. Barak](https://orcid.org/0000-0002-3436-2726) [157] [, E.L. Barberio](https://orcid.org/0000-0002-3111-0910) [101] [, D. Barberis](https://orcid.org/0000-0002-3938-4553) [53b,53a],
[M. Barbero](https://orcid.org/0000-0002-7824-3358) [98], G. Barbour [92] [, K.N. Barends](https://orcid.org/0000-0002-9165-9331) [31a] [, T. Barillari](https://orcid.org/0000-0001-7326-0565) [111] [, M-S. Barisits](https://orcid.org/0000-0003-0253-106X) [34] [, J. Barkeloo](https://orcid.org/0000-0002-5132-4887) [127],
[T. Barklow](https://orcid.org/0000-0002-7709-037X) [149] [, B.M. Barnett](https://orcid.org/0000-0002-5361-2823) [139] [, R.M. Barnett](https://orcid.org/0000-0002-7210-9887) [16] [, A. Baroncelli](https://orcid.org/0000-0001-7090-7474) [58a] [, G. Barone](https://orcid.org/0000-0001-5163-5936) [27] [, A.J. Barr](https://orcid.org/0000-0002-3533-3740) [130],
[L. Barranco Navarro](https://orcid.org/0000-0002-3380-8167) [43a,43b] [, F. Barreiro](https://orcid.org/0000-0002-3021-0258) [95] [, J. Barreiro Guimar˜aes da Costa](https://orcid.org/0000-0003-2387-0386) [13a] [, U. Barron](https://orcid.org/0000-0002-3455-7208) [157] [, S. Barsov](https://orcid.org/0000-0003-2872-7116) [133],
[F. Bartels](https://orcid.org/0000-0002-3407-0918) [59a] [, R. Bartoldus](https://orcid.org/0000-0001-5317-9794) [149] [, G. Bartolini](https://orcid.org/0000-0002-9313-7019) [98] [, A.E. Barton](https://orcid.org/0000-0001-9696-9497) [87] [, P. Bartos](https://orcid.org/0000-0003-1419-3213) [26a] [, A. Basalaev](https://orcid.org/0000-0001-5623-2853) [44] [, A. Basan](https://orcid.org/0000-0001-8021-8525) [96],
[M. Baselga](https://orcid.org/0000-0002-1533-0876) [44] [, I. Bashta](https://orcid.org/0000-0002-2961-2735) [72a,72b] [, A. Bassalat](https://orcid.org/0000-0002-0129-1423) [62] [, M.J. Basso](https://orcid.org/0000-0001-9278-3863) [162] [, C.R. Basson](https://orcid.org/0000-0003-1693-5946) [97] [, R.L. Bates](https://orcid.org/0000-0002-6923-5372) [55], S. Batlamous [33e],
[J.R. Batley](https://orcid.org/0000-0001-7658-7766) [30] [, B. Batool](https://orcid.org/0000-0001-6544-9376) [147], M. Battaglia [141] [, M. Bauce](https://orcid.org/0000-0002-9148-4658) [70a,70b] [, F. Bauer](https://orcid.org/0000-0003-2258-2892) [140,*] [, P. Bauer](https://orcid.org/0000-0002-4568-5360) [22], H.S. Bawa [29],
[A. Bayirli](https://orcid.org/0000-0003-3542-7242) [11c] [, J.B. Beacham](https://orcid.org/0000-0003-3623-3335) [47] [, T. Beau](https://orcid.org/0000-0002-2022-2140) [131] [, P.H. Beauchemin](https://orcid.org/0000-0003-4889-8748) [165] [, F. Becherer](https://orcid.org/0000-0003-0562-4616) [50] [, P. Bechtle](https://orcid.org/0000-0003-3479-2221) [22] [, H.P. Beck](https://orcid.org/0000-0001-7212-1096) [18,q],
[K. Becker](https://orcid.org/0000-0002-6691-6498) [173] [, C. Becot](https://orcid.org/0000-0003-0473-512X) [44] [, A.J. Beddall](https://orcid.org/0000-0002-8451-9672) [11a] [, V.A. Bednyakov](https://orcid.org/0000-0003-4864-8909) [77] [, C.P. Bee](https://orcid.org/0000-0001-6294-6561) [151] [, T.A. Beermann](https://orcid.org/0000-0001-9805-2893) [34] [, M. Begalli](https://orcid.org/0000-0003-4868-6059) [78b],
[M. Begel](https://orcid.org/0000-0002-1634-4399) [27] [, A. Behera](https://orcid.org/0000-0002-7739-295X) [151] [, J.K. Behr](https://orcid.org/0000-0002-5501-4640) [44] [, C. Beirao Da Cruz E Silva](https://orcid.org/0000-0002-1231-3819) [34] [, J.F. Beirer](https://orcid.org/0000-0001-9024-4989) [51,34] [, F. Beisiegel](https://orcid.org/0000-0002-7659-8948) [22],
[M. Belfkir](https://orcid.org/0000-0001-9974-1527) [4] [, G. Bella](https://orcid.org/0000-0002-4009-0990) [157] [, L. Bellagamba](https://orcid.org/0000-0001-7098-9393) [21b] [, A. Bellerive](https://orcid.org/0000-0001-6775-0111) [32] [, P. Bellos](https://orcid.org/0000-0003-2049-9622) [19] [, K. Beloborodov](https://orcid.org/0000-0003-0945-4087) [117b,117a],
[K. Belotskiy](https://orcid.org/0000-0003-4617-8819) [108] [, N.L. Belyaev](https://orcid.org/0000-0002-1131-7121) [108] [, D. Benchekroun](https://orcid.org/0000-0001-5196-8327) [33a] [, Y. Benhammou](https://orcid.org/0000-0002-0392-1783) [157] [, D.P. Benjamin](https://orcid.org/0000-0001-9338-4581) [27] [, M. Benoit](https://orcid.org/0000-0002-8623-1699) [27],
[J.R. Bensinger](https://orcid.org/0000-0002-6117-4536) [24] [, S. Bentvelsen](https://orcid.org/0000-0003-3280-0953) [115] [, L. Beresford](https://orcid.org/0000-0002-3080-1824) [34] [, M. Beretta](https://orcid.org/0000-0002-7026-8171) [49] [, D. Berge](https://orcid.org/0000-0002-2918-1824) [17] [, E. Bergeaas Kuutmann](https://orcid.org/0000-0002-1253-8583) [167],
[N. Berger](https://orcid.org/0000-0002-7963-9725) [4] [, B. Bergmann](https://orcid.org/0000-0002-8076-5614) [137] [, L.J. Bergsten](https://orcid.org/0000-0002-0398-2228) [24] [, J. Beringer](https://orcid.org/0000-0002-9975-1781) [16] [, S. Berlendis](https://orcid.org/0000-0003-1911-772X) [6] [, G. Bernardi](https://orcid.org/0000-0002-2837-2442) [131] [, C. Bernius](https://orcid.org/0000-0003-3433-1687) [149],
[F.U. Bernlochner](https://orcid.org/0000-0001-8153-2719) [22] [, T. Berry](https://orcid.org/0000-0002-9569-8231) [91] [, P. Berta](https://orcid.org/0000-0003-0780-0345) [138] [, A. Berthold](https://orcid.org/0000-0002-3824-409X) [46] [, I.A. Bertram](https://orcid.org/0000-0003-4073-4941) [87] [, O. Bessidskaia Bylund](https://orcid.org/0000-0003-2011-3005) [177],
[S. Bethke](https://orcid.org/0000-0003-0073-3821) [111] [, A. Betti](https://orcid.org/0000-0003-0839-9311) [40] [, A.J. Bevan](https://orcid.org/0000-0002-4105-9629) [90] [, S. Bhatta](https://orcid.org/0000-0002-9045-3278) [151] [, D.S. Bhattacharya](https://orcid.org/0000-0003-3837-4166) [172], P. Bhattarai [24] [, V.S. Bhopatkar](https://orcid.org/0000-0003-3024-587X) [5],
R. Bi [134] [, R.M. Bianchi](https://orcid.org/0000-0001-7345-7798) [134] [, O. Biebel](https://orcid.org/0000-0002-8663-6856) [110] [, R. Bielski](https://orcid.org/0000-0002-2079-5344) [127] [, N.V. Biesuz](https://orcid.org/0000-0003-3004-0946) [69a,69b] [, M. Biglietti](https://orcid.org/0000-0001-5442-1351) [72a],


59


[T.R.V. Billoud](https://orcid.org/0000-0002-6280-3306) [137] [, M. Bindi](https://orcid.org/0000-0001-6172-545X) [51] [, A. Bingul](https://orcid.org/0000-0002-2455-8039) [11d] [, C. Bini](https://orcid.org/0000-0001-6674-7869) [70a,70b] [, S. Biondi](https://orcid.org/0000-0002-1492-6715) [21b,21a] [, A. Biondini](https://orcid.org/0000-0002-1559-3473) [88],
[C.J. Birch-sykes](https://orcid.org/0000-0001-6329-9191) [97] [, G.A. Bird](https://orcid.org/0000-0003-2025-5935) [19,139] [, M. Birman](https://orcid.org/0000-0002-3835-0968) [175], T. Bisanz [34] [, J.P. Biswal](https://orcid.org/0000-0001-8361-2309) [2] [, D. Biswas](https://orcid.org/0000-0002-7543-3471) [176,j] [, A. Bitadze](https://orcid.org/0000-0001-7979-1092) [97],
[C. Bittrich](https://orcid.org/0000-0003-3628-5995) [46] [, K. Bjørke](https://orcid.org/0000-0003-3485-0321) [129] [, I. Bloch](https://orcid.org/0000-0002-6696-5169) [44] [, C. Blocker](https://orcid.org/0000-0001-6898-5633) [24] [, A. Blue](https://orcid.org/0000-0002-7716-5626) [55] [, U. Blumenschein](https://orcid.org/0000-0002-6134-0303) [90] [, J. Blumenthal](https://orcid.org/0000-0001-5412-1236) [96],
[G.J. Bobbink](https://orcid.org/0000-0001-8462-351X) [115] [, V.S. Bobrovnikov](https://orcid.org/0000-0002-2003-0261) [117b,117a] [, M. Boehler](https://orcid.org/0000-0001-9734-574X) [50] [, D. Bogavac](https://orcid.org/0000-0003-2138-9062) [12] [, A.G. Bogdanchikov](https://orcid.org/0000-0002-8635-9342) [117b,117a],
C. Bohm [43a] [, V. Boisvert](https://orcid.org/0000-0002-7736-0173) [91] [, P. Bokan](https://orcid.org/0000-0002-2668-889X) [44] [, T. Bold](https://orcid.org/0000-0002-2432-411X) [81a] [, M. Bomben](https://orcid.org/0000-0002-9807-861X) [131] [, M. Bona](https://orcid.org/0000-0002-9660-580X) [90] [, M. Boonekamp](https://orcid.org/0000-0003-0078-9817) [140],
[C.D. Booth](https://orcid.org/0000-0001-5880-7761) [91] [, A.G. Borbely](https://orcid.org/0000-0002-6890-1601) ´ [55] [, H.M. Borecka-Bielska](https://orcid.org/0000-0002-5702-739X) [106] [, L.S. Borgna](https://orcid.org/0000-0003-0012-7856) [92] [, G. Borissov](https://orcid.org/0000-0002-4226-9521) [87] [, D. Bortoletto](https://orcid.org/0000-0002-1287-4712) [130],
[D. Boscherini](https://orcid.org/0000-0001-9207-6413) [21b] [, M. Bosman](https://orcid.org/0000-0002-7290-643X) [12] [, J.D. Bossio Sola](https://orcid.org/0000-0002-7134-8077) [34] [, K. Bouaouda](https://orcid.org/0000-0002-7723-5030) [33a] [, J. Boudreau](https://orcid.org/0000-0002-9314-5860) [134],
[E.V. Bouhova-Thacker](https://orcid.org/0000-0002-5103-1558) [87] [, D. Boumediene](https://orcid.org/0000-0002-7809-3118) [36] [, R. Bouquet](https://orcid.org/0000-0001-9683-7101) [131] [, A. Boveia](https://orcid.org/0000-0002-6647-6699) [123] [, J. Boyd](https://orcid.org/0000-0001-7360-0726) [34] [, D. Boye](https://orcid.org/0000-0002-2704-835X) [27],
[I.R. Boyko](https://orcid.org/0000-0002-3355-4662) [77] [, A.J. Bozson](https://orcid.org/0000-0003-2354-4812) [91] [, J. Bracinik](https://orcid.org/0000-0001-5762-3477) [19] [, N. Brahimi](https://orcid.org/0000-0003-0992-3509) [58d,58c] [, G. Brandt](https://orcid.org/0000-0001-7992-0309) [177] [, O. Brandt](https://orcid.org/0000-0001-5219-1417) [30] [, F. Braren](https://orcid.org/0000-0003-4339-4727) [44],
[B. Brau](https://orcid.org/0000-0001-9726-4376) [99] [, J.E. Brau](https://orcid.org/0000-0003-1292-9725) [127], W.D. Breaden Madden [55] [, K. Brendlinger](https://orcid.org/0000-0002-9096-780X) [44] [, R. Brener](https://orcid.org/0000-0001-5791-4872) [175] [, L. Brenner](https://orcid.org/0000-0001-5350-7081) [34],
[R. Brenner](https://orcid.org/0000-0002-8204-4124) [167] [, S. Bressler](https://orcid.org/0000-0003-4194-2734) [175] [, B. Brickwedde](https://orcid.org/0000-0003-3518-3057) [96] [, D.L. Briglin](https://orcid.org/0000-0002-3048-8153) [19] [, D. Britton](https://orcid.org/0000-0001-9998-4342) [55] [, D. Britzger](https://orcid.org/0000-0002-9246-7366) [111] [, I. Brock](https://orcid.org/0000-0003-0903-8948) [22],
[R. Brock](https://orcid.org/0000-0002-4556-9212) [103] [, G. Brooijmans](https://orcid.org/0000-0002-3354-1810) [37] [, W.K. Brooks](https://orcid.org/0000-0001-6161-3570) [142e] [, E. Brost](https://orcid.org/0000-0002-6800-9808) [27] [, P.A. Bruckman de Renstrom](https://orcid.org/0000-0002-0206-1160) [82] [, B. Br¨uers](https://orcid.org/0000-0002-1479-2112) [44],
[D. Bruncko](https://orcid.org/0000-0003-0208-2372) [26b] [, A. Bruni](https://orcid.org/0000-0003-4806-0718) [21b] [, G. Bruni](https://orcid.org/0000-0001-5667-7748) [21b] [, M. Bruschi](https://orcid.org/0000-0002-4319-4023) [21b] [, N. Bruscino](https://orcid.org/0000-0002-6168-689X) [70a,70b] [, L. Bryngemark](https://orcid.org/0000-0002-8420-3408) [149],
[T. Buanes](https://orcid.org/0000-0002-8977-121X) [15] [, Q. Buat](https://orcid.org/0000-0001-7318-5251) [151] [, P. Buchholz](https://orcid.org/0000-0002-4049-0134) [147] [, A.G. Buckley](https://orcid.org/0000-0001-8355-9237) [55] [, I.A. Budagov](https://orcid.org/0000-0002-3711-148X) [77] [, M.K. Bugge](https://orcid.org/0000-0002-8650-8125) [129] [, O. Bulekov](https://orcid.org/0000-0002-5687-2073) [108],
[B.A. Bullard](https://orcid.org/0000-0001-7148-6536) [57] [, S. Burdin](https://orcid.org/0000-0003-4831-4132) [88] [, C.D. Burgard](https://orcid.org/0000-0002-6900-825X) [44] [, A.M. Burger](https://orcid.org/0000-0003-0685-4122) [125] [, B. Burghgrave](https://orcid.org/0000-0001-5686-0948) [7] [, J.T.P. Burr](https://orcid.org/0000-0001-6726-6362) [44],
[C.D. Burton](https://orcid.org/0000-0002-3427-6537) [10] [, J.C. Burzynski](https://orcid.org/0000-0002-4690-0528) [148] [, E.L. Busch](https://orcid.org/0000-0003-4482-2666) [37] [, V. B¨uscher](https://orcid.org/0000-0001-9196-0629) [96] [, P.J. Bussey](https://orcid.org/0000-0003-0988-7878) [55] [, J.M. Butler](https://orcid.org/0000-0003-2834-836X) [23] [, C.M. Buttar](https://orcid.org/0000-0003-0188-6491) [55],
[J.M. Butterworth](https://orcid.org/0000-0002-5905-5394) [92] [, W. Buttinger](https://orcid.org/0000-0002-5116-1897) [139], C.J. Buxo Vazquez [103] [, A.R. Buzykaev](https://orcid.org/0000-0002-5458-5564) [117b,117a] [, G. Cabras](https://orcid.org/0000-0002-8467-8235) [21b],
[S. Cabrera Urb´an](https://orcid.org/0000-0001-7640-7913) [169] [, D. Caforio](https://orcid.org/0000-0001-7808-8442) [54] [, H. Cai](https://orcid.org/0000-0001-7575-3603) [134] [, V.M.M. Cairo](https://orcid.org/0000-0002-0758-7575) [149] [, O. Cakir](https://orcid.org/0000-0002-9016-138X) [3a] [, N. Calace](https://orcid.org/0000-0002-1494-9538) [34] [, P. Calafiura](https://orcid.org/0000-0002-1692-1678) [16],
[G. Calderini](https://orcid.org/0000-0002-9495-9145) [131] [, P. Calfayan](https://orcid.org/0000-0003-1600-464X) [63] [, G. Callea](https://orcid.org/0000-0001-5969-3786) [55], L.P. Caloba [78b] [, D. Calvet](https://orcid.org/0000-0002-9953-5333) [36] [, S. Calvet](https://orcid.org/0000-0002-2531-3463) [36] [, T.P. Calvet](https://orcid.org/0000-0002-3342-3566) [98],
[M. Calvetti](https://orcid.org/0000-0003-0125-2165) [69a,69b] [, R. Camacho Toro](https://orcid.org/0000-0002-9192-8028) [131] [, S. Camarda](https://orcid.org/0000-0003-0479-7689) [34] [, D. Camarero Munoz](https://orcid.org/0000-0002-2855-7738) [95] [, P. Camarri](https://orcid.org/0000-0002-5732-5645) [71a,71b],
[M.T. Camerlingo](https://orcid.org/0000-0002-9417-8613) [72a,72b] [, D. Cameron](https://orcid.org/0000-0001-6097-2256) [129] [, C. Camincher](https://orcid.org/0000-0001-5929-1357) [171] [, M. Campanelli](https://orcid.org/0000-0001-6746-3374) [92] [, A. Camplani](https://orcid.org/0000-0002-6386-9788) [38],
[V. Canale](https://orcid.org/0000-0003-2303-9306) [67a,67b] [, A. Canesse](https://orcid.org/0000-0002-9227-5217) [100] [, M. Cano Bret](https://orcid.org/0000-0002-8880-434X) [75] [, J. Cantero](https://orcid.org/0000-0001-8449-1019) [125] [, Y. Cao](https://orcid.org/0000-0001-8747-2809) [168] [, F. Capocasa](https://orcid.org/0000-0002-3562-9592) [24] [, M. Capua](https://orcid.org/0000-0002-2443-6525) [39b,39a],
A. Carbone [66a,66b] [, R. Cardarelli](https://orcid.org/0000-0003-4541-4189) [71a] [, J.C.J. Cardenas](https://orcid.org/0000-0002-6511-7096) [7] [, F. Cardillo](https://orcid.org/0000-0002-4478-3524) [169] [, G. Carducci](https://orcid.org/0000-0002-4376-4911) [39b,39a] [, T. Carli](https://orcid.org/0000-0003-4058-5376) [34],
[G. Carlino](https://orcid.org/0000-0002-3924-0445) [67a] [, B.T. Carlson](https://orcid.org/0000-0002-7550-7821) [134] [, E.M. Carlson](https://orcid.org/0000-0002-4139-9543) [171,163a] [, L. Carminati](https://orcid.org/0000-0003-4535-2926) [66a,66b] [, M. Carnesale](https://orcid.org/0000-0003-3570-7332) [70a,70b],
[R.M.D. Carney](https://orcid.org/0000-0001-5659-4440) [149] [, S. Caron](https://orcid.org/0000-0003-2941-2829) [114] [, E. Carquin](https://orcid.org/0000-0002-7863-1166) [142e] [, S. Carr´a](https://orcid.org/0000-0001-8650-942X) [44] [, G. Carratta](https://orcid.org/0000-0002-8846-2714) [21b,21a] [, J.W.S. Carter](https://orcid.org/0000-0002-7836-4264) [162],
[T.M. Carter](https://orcid.org/0000-0003-2966-6036) [48] [, D. Casadei](https://orcid.org/0000-0002-3343-3529) [31c] [, M.P. Casado](https://orcid.org/0000-0002-0394-5646) [12,g], A.F. Casha [162] [, E.G. Castiglia](https://orcid.org/0000-0001-7991-2018) [178] [, F.L. Castillo](https://orcid.org/0000-0002-1172-1052) [59a],
[L. Castillo Garcia](https://orcid.org/0000-0003-1396-2826) [12] [, V. Castillo Gimenez](https://orcid.org/0000-0002-8245-1790) [169] [, N.F. Castro](https://orcid.org/0000-0001-8491-4376) [135a,135e] [, A. Catinaccio](https://orcid.org/0000-0001-8774-8887) [34] [, J.R. Catmore](https://orcid.org/0000-0001-8915-0184) [129],
A. Cattai [34] [, V. Cavaliere](https://orcid.org/0000-0002-4297-8539) [27] [, N. Cavalli](https://orcid.org/0000-0002-1096-5290) [21b,21a] [, V. Cavasinni](https://orcid.org/0000-0001-6203-9347) [69a,69b] [, E. Celebi](https://orcid.org/0000-0003-3793-0159) [11b] [, F. Celli](https://orcid.org/0000-0001-6962-4573) [130],
[M.S. Centonze](https://orcid.org/0000-0002-7945-4392) [65a,65b] [, K. Cerny](https://orcid.org/0000-0003-0683-2177) [126] [, A.S. Cerqueira](https://orcid.org/0000-0002-4300-703X) [78a] [, A. Cerri](https://orcid.org/0000-0002-1904-6661) [152] [, L. Cerrito](https://orcid.org/0000-0002-8077-7850) [71a,71b] [, F. Cerutti](https://orcid.org/0000-0001-9669-9642) [16],
[A. Cervelli](https://orcid.org/0000-0002-0518-1459) [21b] [, S.A. Cetin](https://orcid.org/0000-0001-5050-8441) [11b] [, Z. Chadi](https://orcid.org/0000-0002-3117-5415) [33a] [, D. Chakraborty](https://orcid.org/0000-0002-9865-4146) [116] [, M. Chala](https://orcid.org/0000-0002-4343-9094) [135f] [, J. Chan](https://orcid.org/0000-0001-7069-0295) [176] [, W.S. Chan](https://orcid.org/0000-0003-2150-1296) [115],
[W.Y. Chan](https://orcid.org/0000-0002-5369-8540) [88] [, J.D. Chapman](https://orcid.org/0000-0002-2926-8962) [30] [, B. Chargeishvili](https://orcid.org/0000-0002-5376-2397) [155b] [, D.G. Charlton](https://orcid.org/0000-0003-0211-2041) [19] [, T.P. Charman](https://orcid.org/0000-0001-6288-5236) [90] [, M. Chatterjee](https://orcid.org/0000-0003-4241-7405) [18],

[S. Chekanov](https://orcid.org/0000-0001-7314-7247) [5] [, S.V. Chekulaev](https://orcid.org/0000-0002-4034-2326) [163a] [, G.A. Chelkov](https://orcid.org/0000-0002-3468-9761) [77,ae] [, A. Chen](https://orcid.org/0000-0001-9973-7966) [102] [, B. Chen](https://orcid.org/0000-0002-3034-8943) [157] [, B. Chen](https://orcid.org/0000-0002-7985-9023) [171], C. Chen [58a],
[C.H. Chen](https://orcid.org/0000-0003-1589-9955) [76] [, H. Chen](https://orcid.org/0000-0002-5895-6799) [13c] [, H. Chen](https://orcid.org/0000-0002-9936-0115) [27] [, J. Chen](https://orcid.org/0000-0002-2554-2725) [58c] [, J. Chen](https://orcid.org/0000-0003-1586-5253) [24] [, S. Chen](https://orcid.org/0000-0001-7987-9764) [132] [, S.J. Chen](https://orcid.org/0000-0003-0447-5348) [13c] [, X. Chen](https://orcid.org/0000-0003-4977-2717) [58c],
[X. Chen](https://orcid.org/0000-0003-4027-3305) [13b] [, Y. Chen](https://orcid.org/0000-0001-6793-3604) [58a] [, Y-H. Chen](https://orcid.org/0000-0002-2720-1115) [44] [, C.L. Cheng](https://orcid.org/0000-0002-4086-1847) [176] [, H.C. Cheng](https://orcid.org/0000-0002-8912-4389) [60a] [, A. Cheplakov](https://orcid.org/0000-0002-0967-2351) [77],
[E. Cheremushkina](https://orcid.org/0000-0002-8772-0961) [44] [, E. Cherepanova](https://orcid.org/0000-0002-3150-8478) [77] [, R. Cherkaoui El Moursli](https://orcid.org/0000-0002-5842-2818) [33e] [, E. Cheu](https://orcid.org/0000-0002-2562-9724) [6] [, K. Cheung](https://orcid.org/0000-0003-2176-4053) [61],
[L. Chevalier](https://orcid.org/0000-0003-3762-7264) [140] [, V. Chiarella](https://orcid.org/0000-0002-4210-2924) [49] [, G. Chiarelli](https://orcid.org/0000-0001-9851-4816) [69a] [, G. Chiodini](https://orcid.org/0000-0002-2458-9513) [65a] [, A.S. Chisholm](https://orcid.org/0000-0001-9214-8528) [19] [, A. Chitan](https://orcid.org/0000-0003-2262-4773) [25b],
[Y.H. Chiu](https://orcid.org/0000-0002-9487-9348) [171] [, M.V. Chizhov](https://orcid.org/0000-0001-5841-3316) [77,s] [, K. Choi](https://orcid.org/0000-0003-0748-694X) [10] [, A.R. Chomont](https://orcid.org/0000-0002-3243-5610) [70a,70b] [, Y. Chou](https://orcid.org/0000-0002-2204-5731) [99], Y.S. Chow [115],
[T. Chowdhury](https://orcid.org/0000-0002-2681-8105) [31f] [, L.D. Christopher](https://orcid.org/0000-0002-2509-0132) [31f] [, M.C. Chu](https://orcid.org/0000-0002-1971-0403) [60a] [, X. Chu](https://orcid.org/0000-0003-2848-0184) [13a,13d] [, J. Chudoba](https://orcid.org/0000-0002-6425-2579) [136] [, J.J. Chwastowski](https://orcid.org/0000-0002-6190-8376) [82],
[D. Cieri](https://orcid.org/0000-0002-3533-3847) [111] [, K.M. Ciesla](https://orcid.org/0000-0003-2751-3474) [82] [, V. Cindro](https://orcid.org/0000-0002-2037-7185) [89] [, I.A. Cioar˘a](https://orcid.org/0000-0002-9224-3784) [25b] [, A. Ciocio](https://orcid.org/0000-0002-3081-4879) [16] [, F. Cirotto](https://orcid.org/0000-0001-6556-856X) [67a,67b] [, Z.H. Citron](https://orcid.org/0000-0003-1831-6452) [175,k],
[M. Citterio](https://orcid.org/0000-0002-0842-0654) [66a], D.A. Ciubotaru [25b] [, B.M. Ciungu](https://orcid.org/0000-0002-8920-4880) [162] [, A. Clark](https://orcid.org/0000-0001-8341-5911) [52] [, P.J. Clark](https://orcid.org/0000-0002-3777-0880) [48] [, J.M. Clavijo Columbie](https://orcid.org/0000-0003-3210-1722) [44],
[S.E. Clawson](https://orcid.org/0000-0001-9952-934X) [97] [, C. Clement](https://orcid.org/0000-0003-3122-3605) [43a,43b] [, L. Clissa](https://orcid.org/0000-0002-4876-5200) [21b,21a] [, Y. Coadou](https://orcid.org/0000-0001-8195-7004) [98] [, M. Cobal](https://orcid.org/0000-0003-3309-0762) [64a,64c] [, A. Coccaro](https://orcid.org/0000-0003-2368-4559) [53b],
J. Cochran [76] [, R.F. Coelho Barrue](https://orcid.org/0000-0001-8985-5379) [135a] [, R. Coelho Lopes De Sa](https://orcid.org/0000-0001-5200-9195) [99] [, S. Coelli](https://orcid.org/0000-0002-5145-3646) [66a], H. Cohen [157],
[A.E.C. Coimbra](https://orcid.org/0000-0003-2301-1637) [34] [, B. Cole](https://orcid.org/0000-0002-5092-2148) [37] [, J. Collot](https://orcid.org/0000-0002-9412-7090) [56] [, P. Conde Mui˜no](https://orcid.org/0000-0002-9187-7478) [135a,135g] [, S.H. Connell](https://orcid.org/0000-0001-6000-7245) [31c] [, I.A. Connelly](https://orcid.org/0000-0001-9127-6827) [55],
[E.I. Conroy](https://orcid.org/0000-0002-0215-2767) [130] [, F. Conventi](https://orcid.org/0000-0002-5575-1413) [67a,aj] [, H.G. Cooke](https://orcid.org/0000-0001-9297-1063) [19] [, A.M. Cooper-Sarkar](https://orcid.org/0000-0002-7107-5902) [130] [, F. Cormier](https://orcid.org/0000-0002-2532-3207) [170] [, L.D. Corpe](https://orcid.org/0000-0003-2136-4842) [34],
[M. Corradi](https://orcid.org/0000-0001-8729-466X) [70a,70b] [, E.E. Corrigan](https://orcid.org/0000-0003-2485-0248) [94] [, F. Corriveau](https://orcid.org/0000-0002-4970-7600) [100,y] [, M.J. Costa](https://orcid.org/0000-0002-2064-2954) [169] [, F. Costanza](https://orcid.org/0000-0002-8056-8469) [4] [, D. Costanzo](https://orcid.org/0000-0003-4920-6264) [145],
[B.M. Cote](https://orcid.org/0000-0003-2444-8267) [123] [, G. Cowan](https://orcid.org/0000-0001-8363-9827) [91] [, J.W. Cowley](https://orcid.org/0000-0001-7002-652X) [30] [, K. Cranmer](https://orcid.org/0000-0002-5769-7094) [121] [, S. Cr´ep´e-Renaudin](https://orcid.org/0000-0001-5980-5805) [56] [, F. Crescioli](https://orcid.org/0000-0001-6457-2575) [131],
[M. Cristinziani](https://orcid.org/0000-0003-3893-9171) [147] [, M. Cristoforetti](https://orcid.org/0000-0002-0127-1342) [73a,73b,b] [, V. Croft](https://orcid.org/0000-0002-8731-4525) [165] [, G. Crosetti](https://orcid.org/0000-0001-5990-4811) [39b,39a] [, A. Cueto](https://orcid.org/0000-0003-1494-7898) [34],


60


[T. Cuhadar Donszelmann](https://orcid.org/0000-0003-3519-1356) [166] [, H. Cui](https://orcid.org/0000-0002-9923-1313) [13a,13d] [, A.R. Cukierman](https://orcid.org/0000-0002-7834-1716) [149] [, W.R. Cunningham](https://orcid.org/0000-0001-5517-8795) [55] [, F. Curcio](https://orcid.org/0000-0002-8682-9316) [39b,39a],
[P. Czodrowski](https://orcid.org/0000-0003-0723-1437) [34] [, M.M. Czurylo](https://orcid.org/0000-0003-1943-5883) [59b] [, M.J. Da Cunha Sargedas De Sousa](https://orcid.org/0000-0001-7991-593X) [58a] [, J.V. Da Fonseca Pinto](https://orcid.org/0000-0003-1746-1914) [78b],
[C. Da Via](https://orcid.org/0000-0001-6154-7323) [97] [, W. Dabrowski](https://orcid.org/0000-0001-9061-9568) [81a] [, T. Dado](https://orcid.org/0000-0002-7050-2669) [45] [, S. Dahbi](https://orcid.org/0000-0002-5222-7894) [31f] [, T. Dai](https://orcid.org/0000-0002-9607-5124) [102] [, C. Dallapiccola](https://orcid.org/0000-0002-1391-2477) [99] [, M. Dam](https://orcid.org/0000-0001-6278-9674) [38],
[G. D’amen](https://orcid.org/0000-0002-9742-3709) [27] [, V. D’Amico](https://orcid.org/0000-0002-2081-0129) [72a,72b] [, J. Damp](https://orcid.org/0000-0002-7290-1372) [96] [, J.R. Dandoy](https://orcid.org/0000-0002-9271-7126) [132] [, M.F. Daneri](https://orcid.org/0000-0002-2335-793X) [28] [, M. Danninger](https://orcid.org/0000-0002-7807-7484) [148] [, V. Dao](https://orcid.org/0000-0003-1645-8393) [34],
[G. Darbo](https://orcid.org/0000-0003-2165-0638) [53b] [, S. Darmora](https://orcid.org/0000-0002-9766-3657) [5] [, A. Dattagupta](https://orcid.org/0000-0002-1559-9525) [127] [, S. D’Auria](https://orcid.org/0000-0003-3393-6318) [66a,66b] [, C. David](https://orcid.org/0000-0002-1794-1443) [163b] [, T. Davidek](https://orcid.org/0000-0002-3770-8307) [138] [, D.R. Davis](https://orcid.org/0000-0003-2679-1288) [47],
[B. Davis-Purcell](https://orcid.org/0000-0002-4544-169X) [32] [, I. Dawson](https://orcid.org/0000-0002-5177-8950) [90] [, K. De](https://orcid.org/0000-0002-5647-4489) [7] [, R. De Asmundis](https://orcid.org/0000-0002-7268-8401) [67a] [, M. De Beurs](https://orcid.org/0000-0002-4285-2047) [115] [, S. De Castro](https://orcid.org/0000-0003-2178-5620) [21b,21a],
[N. De Groot](https://orcid.org/0000-0001-6850-4078) [114] [, P. de Jong](https://orcid.org/0000-0002-5330-2614) [115] [, H. De la Torre](https://orcid.org/0000-0002-4516-5269) [103] [, A. De Maria](https://orcid.org/0000-0001-6651-845X) [13c] [, D. De Pedis](https://orcid.org/0000-0002-8151-581X) [70a] [, A. De Salvo](https://orcid.org/0000-0001-8099-7821) [70a],
[U. De Sanctis](https://orcid.org/0000-0003-4704-525X) [71a,71b] [, M. De Santis](https://orcid.org/0000-0001-6423-0719) [71a,71b] [, A. De Santo](https://orcid.org/0000-0002-9158-6646) [152] [, J.B. De Vivie De Regie](https://orcid.org/0000-0001-9163-2211) [56], D.V. Dedovich [77],
[J. Degens](https://orcid.org/0000-0002-6966-4935) [115] [, A.M. Deiana](https://orcid.org/0000-0003-0360-6051) [40] [, J. Del Peso](https://orcid.org/0000-0001-7090-4134) [95] [, Y. Delabat Diaz](https://orcid.org/0000-0002-6096-7649) [44] [, F. Deliot](https://orcid.org/0000-0003-0777-6031) [140] [, C.M. Delitzsch](https://orcid.org/0000-0001-7021-3333) [6],
[M. Della Pietra](https://orcid.org/0000-0003-4446-3368) [67a,67b] [, D. Della Volpe](https://orcid.org/0000-0001-8530-7447) [52] [, A. Dell’Acqua](https://orcid.org/0000-0003-2453-7745) [34] [, L. Dell’Asta](https://orcid.org/0000-0002-9601-4225) [66a,66b] [, M. Delmastro](https://orcid.org/0000-0003-2992-3805) [4],
[P.A. Delsart](https://orcid.org/0000-0002-9556-2924) [56] [, S. Demers](https://orcid.org/0000-0002-7282-1786) [178] [, M. Demichev](https://orcid.org/0000-0002-7730-3072) [77] [, S.P. Denisov](https://orcid.org/0000-0002-4028-7881) [118] [, L. D’Eramo](https://orcid.org/0000-0002-4910-5378) [116] [, D. Derendarz](https://orcid.org/0000-0001-5660-3095) [82],
[J.E. Derkaoui](https://orcid.org/0000-0002-7116-8551) [33d] [, F. Derue](https://orcid.org/0000-0002-3505-3503) [131] [, P. Dervan](https://orcid.org/0000-0003-3929-8046) [88] [, K. Desch](https://orcid.org/0000-0001-5836-6118) [22] [, K. Dette](https://orcid.org/0000-0002-9593-6201) [162] [, C. Deutsch](https://orcid.org/0000-0002-6477-764X) [22] [, P.O. Deviveiros](https://orcid.org/0000-0002-8906-5884) [34],
[F.A. Di Bello](https://orcid.org/0000-0002-9870-2021) [70a,70b] [, A. Di Ciaccio](https://orcid.org/0000-0001-8289-5183) [71a,71b] [, L. Di Ciaccio](https://orcid.org/0000-0003-0751-8083) [4] [, A. Di Domenico](https://orcid.org/0000-0001-8078-2759) [70a,70b] [, C. Di Donato](https://orcid.org/0000-0003-2213-9284) [67a,67b],
[A. Di Girolamo](https://orcid.org/0000-0002-9508-4256) [34] [, G. Di Gregorio](https://orcid.org/0000-0002-7838-576X) [69a,69b] [, A. Di Luca](https://orcid.org/0000-0002-9074-2133) [73a,73b] [, B. Di Micco](https://orcid.org/0000-0002-4067-1592) [72a,72b] [, R. Di Nardo](https://orcid.org/0000-0003-1111-3783) [72a,72b],
[C. Diaconu](https://orcid.org/0000-0002-6193-5091) [98] [, F.A. Dias](https://orcid.org/0000-0001-6882-5402) [115] [, T. Dias Do Vale](https://orcid.org/0000-0001-8855-3520) [135a] [, M.A. Diaz](https://orcid.org/0000-0003-1258-8684) [142a] [, F.G. Diaz Capriles](https://orcid.org/0000-0001-7934-3046) [22] [, J. Dickinson](https://orcid.org/0000-0001-5450-5328) [16],
[M. Didenko](https://orcid.org/0000-0001-9942-6543) [169] [, E.B. Diehl](https://orcid.org/0000-0002-7611-355X) [102] [, J. Dietrich](https://orcid.org/0000-0001-7061-1585) [17] [, S. D´ıez Cornell](https://orcid.org/0000-0003-3694-6167) [44] [, C. Diez Pardos](https://orcid.org/0000-0002-0482-1127) [147] [, A. Dimitrievska](https://orcid.org/0000-0003-0086-0599) [16],
[W. Ding](https://orcid.org/0000-0002-4614-956X) [13b] [, J. Dingfelder](https://orcid.org/0000-0001-5767-2121) [22] [, I-M. Dinu](https://orcid.org/0000-0002-2683-7349) [25b] [, S.J. Dittmeier](https://orcid.org/0000-0002-5172-7520) [59b] [, F. Dittus](https://orcid.org/0000-0002-1760-8237) [34] [, F. Djama](https://orcid.org/0000-0003-1881-3360) [98] [, T. Djobava](https://orcid.org/0000-0002-9414-8350) [155b],
[J.I. Djuvsland](https://orcid.org/0000-0002-6488-8219) [15] [, M.A.B. Do Vale](https://orcid.org/0000-0002-0836-6483) [143] [, D. Dodsworth](https://orcid.org/0000-0002-6720-9883) [24] [, C. Doglioni](https://orcid.org/0000-0002-1509-0390) [94] [, J. Dolejsi](https://orcid.org/0000-0001-5821-7067) [138] [, Z. Dolezal](https://orcid.org/0000-0002-5662-3675) [138],
[M. Donadelli](https://orcid.org/0000-0001-8329-4240) [78c] [, B. Dong](https://orcid.org/0000-0002-6075-0191) [58c] [, J. Donini](https://orcid.org/0000-0002-8998-0839) [36] [, A. D’onofrio](https://orcid.org/0000-0002-0343-6331) [13c] [, M. D’Onofrio](https://orcid.org/0000-0003-2408-5099) [88] [, J. Dopke](https://orcid.org/0000-0002-0683-9910) [139] [, A. Doria](https://orcid.org/0000-0002-5381-2649) [67a],
[M.T. Dova](https://orcid.org/0000-0001-6113-0878) [86] [, A.T. Doyle](https://orcid.org/0000-0001-6322-6195) [55] [, E. Drechsler](https://orcid.org/0000-0002-8773-7640) [148] [, E. Dreyer](https://orcid.org/0000-0001-8955-9510) [148] [, T. Dreyer](https://orcid.org/0000-0002-7465-7887) [51] [, A.S. Drobac](https://orcid.org/0000-0003-4782-4034) [165] [, D. Du](https://orcid.org/0000-0002-6758-0113) [58a],
[T.A. du Pree](https://orcid.org/0000-0001-8703-7938) [115] [, F. Dubinin](https://orcid.org/0000-0003-2182-2727) [107] [, M. Dubovsky](https://orcid.org/0000-0002-3847-0775) [26a] [, A. Dubreuil](https://orcid.org/0000-0001-6161-8793) [52] [, E. Duchovni](https://orcid.org/0000-0002-7276-6342) [175] [, G. Duckeck](https://orcid.org/0000-0002-7756-7801) [110],
[O.A. Ducu](https://orcid.org/0000-0001-5914-0524) [34,25b] [, D. Duda](https://orcid.org/0000-0002-5916-3467) [111] [, A. Dudarev](https://orcid.org/0000-0002-8713-8162) [34] [, M. D’uffizi](https://orcid.org/0000-0003-2499-1649) [97] [, L. Duflot](https://orcid.org/0000-0002-4871-2176) [62] [, M. D¨uhrssen](https://orcid.org/0000-0002-5833-7058) [34] [, C. D¨ulsen](https://orcid.org/0000-0003-4813-8757) [177],
[A.E. Dumitriu](https://orcid.org/0000-0003-3310-4642) [25b] [, M. Dunford](https://orcid.org/0000-0002-7667-260X) [59a] [, S. Dungs](https://orcid.org/0000-0001-9935-6397) [45] [, K. Dunne](https://orcid.org/0000-0003-2626-2247) [43a,43b] [, A. Duperrin](https://orcid.org/0000-0002-5789-9825) [98] [, H. Duran Yildiz](https://orcid.org/0000-0003-3469-6045) [3a],
[M. D¨uren](https://orcid.org/0000-0002-6066-4744) [54] [, A. Durglishvili](https://orcid.org/0000-0003-4157-592X) [155b] [, B. Dutta](https://orcid.org/0000-0001-7277-0440) [44] [, G.I. Dyckes](https://orcid.org/0000-0003-1464-0335) [16] [, M. Dyndal](https://orcid.org/0000-0001-9632-6352) [81a] [, S. Dysch](https://orcid.org/0000-0002-7412-9187) [97] [, B.S. Dziedzic](https://orcid.org/0000-0002-0805-9184) [82],
[B. Eckerova](https://orcid.org/0000-0003-0336-3723) [26a], M.G. Eggleston [47] [, E. Egidio Purcino De Souza](https://orcid.org/0000-0001-5370-8377) [78b] [, L.F. Ehrke](https://orcid.org/0000-0002-2701-968X) [52] [, T. Eifert](https://orcid.org/0000-0002-7535-6058) [7] [, G. Eigen](https://orcid.org/0000-0003-3529-5171) [15],
[K. Einsweiler](https://orcid.org/0000-0002-4391-9100) [16] [, T. Ekelof](https://orcid.org/0000-0002-7341-9115) [167] [, Y. El Ghazali](https://orcid.org/0000-0001-9172-2946) [33b] [, H. El Jarrari](https://orcid.org/0000-0002-8955-9681) [33e] [, A. El Moussaouy](https://orcid.org/0000-0002-9669-5374) [33a] [, V. Ellajosyula](https://orcid.org/0000-0001-5997-3569) [167],
[M. Ellert](https://orcid.org/0000-0001-5265-3175) [167] [, F. Ellinghaus](https://orcid.org/0000-0003-3596-5331) [177] [, A.A. Elliot](https://orcid.org/0000-0003-0921-0314) [90] [, N. Ellis](https://orcid.org/0000-0002-1920-4930) [34] [, J. Elmsheuser](https://orcid.org/0000-0001-8899-051X) [27] [, M. Elsing](https://orcid.org/0000-0002-1213-0545) [34] [, D. Emeliyanov](https://orcid.org/0000-0002-1363-9175) [139],
[A. Emerman](https://orcid.org/0000-0003-4963-1148) [37] [, Y. Enari](https://orcid.org/0000-0002-9916-3349) [159] [, J. Erdmann](https://orcid.org/0000-0002-8073-2740) [45] [, A. Ereditato](https://orcid.org/0000-0002-5423-8079) [18] [, P.A. Erland](https://orcid.org/0000-0003-4543-6599) [82] [, M. Errenst](https://orcid.org/0000-0003-4656-3936) [177] [, M. Escalier](https://orcid.org/0000-0003-4270-2775) [62],
[C. Escobar](https://orcid.org/0000-0003-4442-4537) [169] [, O. Estrada Pastor](https://orcid.org/0000-0001-8210-1064) [169] [, E. Etzion](https://orcid.org/0000-0001-6871-7794) [157] [, G. Evans](https://orcid.org/0000-0003-0434-6925) [135a] [, H. Evans](https://orcid.org/0000-0003-2183-3127) [63] [, M.O. Evans](https://orcid.org/0000-0002-4259-018X) [152] [, A. Ezhilov](https://orcid.org/0000-0002-7520-293X) [133],
[F. Fabbri](https://orcid.org/0000-0001-8474-0978) [55] [, L. Fabbri](https://orcid.org/0000-0002-4002-8353) [21b,21a] [, G. Facini](https://orcid.org/0000-0002-4056-4578) [173] [, V. Fadeyev](https://orcid.org/0000-0003-0154-4328) [141] [, R.M. Fakhrutdinov](https://orcid.org/0000-0001-7882-2125) [118] [, S. Falciano](https://orcid.org/0000-0002-7118-341X) [70a],
[P.J. Falke](https://orcid.org/0000-0002-2004-476X) [22] [, S. Falke](https://orcid.org/0000-0002-0264-1632) [34] [, J. Faltova](https://orcid.org/0000-0003-4278-7182) [138] [, Y. Fan](https://orcid.org/0000-0001-7868-3858) [13a] [, Y. Fang](https://orcid.org/0000-0001-8630-6585) [13a] [, G. Fanourakis](https://orcid.org/0000-0001-6689-4957) [42] [, M. Fanti](https://orcid.org/0000-0002-8773-145X) [66a,66b] [, M. Faraj](https://orcid.org/0000-0001-9442-7598) [58c],
[A. Farbin](https://orcid.org/0000-0003-0000-2439) [7] [, A. Farilla](https://orcid.org/0000-0002-3983-0728) [72a] [, E.M. Farina](https://orcid.org/0000-0003-3037-9288) [68a,68b] [, T. Farooque](https://orcid.org/0000-0003-1363-9324) [103] [, S.M. Farrington](https://orcid.org/0000-0001-5350-9271) [48] [, P. Farthouat](https://orcid.org/0000-0002-4779-5432) [34] [, F. Fassi](https://orcid.org/0000-0002-6423-7213) [33e],
[D. Fassouliotis](https://orcid.org/0000-0003-1289-2141) [8] [, M. Faucci Giannelli](https://orcid.org/0000-0003-3731-820X) [71a,71b] [, W.J. Fawcett](https://orcid.org/0000-0003-2596-8264) [30] [, L. Fayard](https://orcid.org/0000-0002-2190-9091) [62] [, O.L. Fedin](https://orcid.org/0000-0002-1733-7158) [133,p] [, M. Feickert](https://orcid.org/0000-0003-4124-7862) [168],
[L. Feligioni](https://orcid.org/0000-0002-1403-0951) [98] [, A. Fell](https://orcid.org/0000-0003-2101-1879) [145] [, C. Feng](https://orcid.org/0000-0001-9138-3200) [58b] [, M. Feng](https://orcid.org/0000-0002-0698-1482) [13b] [, M.J. Fenton](https://orcid.org/0000-0003-1002-6880) [166], A.B. Fenyuk [118] [, S.W. Ferguson](https://orcid.org/0000-0003-1328-4367) [41],
[J. Ferrando](https://orcid.org/0000-0002-1007-7816) [44] [, A. Ferrari](https://orcid.org/0000-0003-2887-5311) [167] [, P. Ferrari](https://orcid.org/0000-0002-1387-153X) [115] [, R. Ferrari](https://orcid.org/0000-0001-5566-1373) [68a] [, D. Ferrere](https://orcid.org/0000-0002-5687-9240) [52] [, C. Ferretti](https://orcid.org/0000-0002-5562-7893) [102], D. Fiacco [48],
[F. Fiedler](https://orcid.org/0000-0002-4610-5612) [96] [, A. Filipˇciˇc](https://orcid.org/0000-0001-5671-1555) [89] [, F. Filthaut](https://orcid.org/0000-0003-3338-2247) [114] [, M.C.N. Fiolhais](https://orcid.org/0000-0001-9035-0335) [135a,135c,a] [, L. Fiorini](https://orcid.org/0000-0002-5070-2735) [169] [, F. Fischer](https://orcid.org/0000-0001-9799-5232) [147],
[W.C. Fisher](https://orcid.org/0000-0003-3043-3045) [103] [, T. Fitschen](https://orcid.org/0000-0002-1152-7372) [19] [, I. Fleck](https://orcid.org/0000-0003-1461-8648) [147] [, P. Fleischmann](https://orcid.org/0000-0001-6968-340X) [102] [, T. Flick](https://orcid.org/0000-0002-8356-6987) [177] [, B.M. Flierl](https://orcid.org/0000-0002-1098-6446) [110] [, L. Flores](https://orcid.org/0000-0002-2748-758X) [132],

[M. Flores](https://orcid.org/0000-0002-4462-2851) [31d] [, L.R. Flores Castillo](https://orcid.org/0000-0003-1551-5974) [60a] [, F.M. Follega](https://orcid.org/0000-0003-2317-9560) [73a,73b] [, N. Fomin](https://orcid.org/0000-0001-9457-394X) [15] [, J.H. Foo](https://orcid.org/0000-0003-4577-0685) [162], B.C. Forland [63],
[A. Formica](https://orcid.org/0000-0001-8308-2643) [140] [, F.A. F¨orster](https://orcid.org/0000-0002-3727-8781) [12] [, A.C. Forti](https://orcid.org/0000-0002-0532-7921) [97], E. Fortin [98] [, M.G. Foti](https://orcid.org/0000-0002-0976-7246) [130] [, L. Fountas](https://orcid.org/0000-0002-9986-6597) [8] [, D. Fournier](https://orcid.org/0000-0003-4836-0358) [62],
[H. Fox](https://orcid.org/0000-0003-3089-6090) [87] [, P. Francavilla](https://orcid.org/0000-0003-1164-6870) [69a,69b] [, S. Francescato](https://orcid.org/0000-0001-5315-9275) [57] [, M. Franchini](https://orcid.org/0000-0002-4554-252X) [21b,21a] [, S. Franchino](https://orcid.org/0000-0002-8159-8010) [59a], D. Francis [34],
[L. Franco](https://orcid.org/0000-0002-1687-4314) [4] [, L. Franconi](https://orcid.org/0000-0002-0647-6072) [18] [, M. Franklin](https://orcid.org/0000-0002-6595-883X) [57] [, G. Frattari](https://orcid.org/0000-0002-7829-6564) [70a,70b] [, A.C. Freegard](https://orcid.org/0000-0003-4482-3001) [90], P.M. Freeman [19],
[W.S. Freund](https://orcid.org/0000-0003-4473-1027) [78b] [, E.M. Freundlich](https://orcid.org/0000-0003-0907-392X) [45] [, D. Froidevaux](https://orcid.org/0000-0003-3986-3922) [34] [, J.A. Frost](https://orcid.org/0000-0003-3562-9944) [130] [, Y. Fu](https://orcid.org/0000-0002-7370-7395) [58a] [, M. Fujimoto](https://orcid.org/0000-0002-6701-8198) [122],

[E. Fullana Torregrosa](https://orcid.org/0000-0003-3082-621X) [169] [, J. Fuster](https://orcid.org/0000-0002-1290-2031) [169] [, A. Gabrielli](https://orcid.org/0000-0001-5346-7841) [21b,21a] [, A. Gabrielli](https://orcid.org/0000-0003-0768-9325) [34] [, P. Gadow](https://orcid.org/0000-0003-4475-6734) [44] [, G. Gagliardi](https://orcid.org/0000-0002-3550-4124) [53b,53a],
[L.G. Gagnon](https://orcid.org/0000-0003-3000-8479) [16] [, G.E. Gallardo](https://orcid.org/0000-0001-5832-5746) [130] [, E.J. Gallas](https://orcid.org/0000-0002-1259-1034) [130] [, B.J. Gallop](https://orcid.org/0000-0001-7401-5043) [139] [, R. Gamboa Goni](https://orcid.org/0000-0003-1026-7633) [90] [, K.K. Gan](https://orcid.org/0000-0002-1550-1487) [123],
[S. Ganguly](https://orcid.org/0000-0003-1285-9261) [159] [, J. Gao](https://orcid.org/0000-0002-8420-3803) [58a] [, Y. Gao](https://orcid.org/0000-0001-6326-4773) [48] [, Y.S. Gao](https://orcid.org/0000-0002-6082-9190) [29,m] [, F.M. Garay Walls](https://orcid.org/0000-0002-6670-1104) [142a] [, C. Garc´ıa](https://orcid.org/0000-0003-1625-7452) [169],
[J.E. Garc´ıa Navarro](https://orcid.org/0000-0002-0279-0523) [169] [, J.A. Garc´ıa Pascual](https://orcid.org/0000-0002-7399-7353) [13a] [, M. Garcia-Sciveres](https://orcid.org/0000-0002-5800-4210) [16] [, R.W. Gardner](https://orcid.org/0000-0003-1433-9366) [35] [, D. Garg](https://orcid.org/0000-0001-8383-9343) [75],
[R.B. Garg](https://orcid.org/0000-0002-2691-7963) [149] [, S. Gargiulo](https://orcid.org/0000-0003-4850-1122) [50], C.A. Garner [162] [, V. Garonne](https://orcid.org/0000-0001-7169-9160) [129] [, S.J. Gasiorowski](https://orcid.org/0000-0002-4067-2472) [144] [, P. Gaspar](https://orcid.org/0000-0002-9232-1332) [78b],


61


[G. Gaudio](https://orcid.org/0000-0002-6833-0933) [68a] [, P. Gauzzi](https://orcid.org/0000-0003-4841-5822) [70a,70b] [, I.L. Gavrilenko](https://orcid.org/0000-0001-7219-2636) [107] [, A. Gavrilyuk](https://orcid.org/0000-0003-3837-6567) [119] [, C. Gay](https://orcid.org/0000-0002-9354-9507) [170] [, G. Gaycken](https://orcid.org/0000-0002-2941-9257) [44] [, E.N. Gazis](https://orcid.org/0000-0002-9272-4254) [9],
[A.A. Geanta](https://orcid.org/0000-0003-2781-2933) [25b] [, C.M. Gee](https://orcid.org/0000-0002-3271-7861) [141] [, C.N.P. Gee](https://orcid.org/0000-0002-8833-3154) [139] [, J. Geisen](https://orcid.org/0000-0003-4644-2472) [94] [, M. Geisen](https://orcid.org/0000-0003-0932-0230) [96] [, C. Gemme](https://orcid.org/0000-0002-1702-5699) [53b] [, M.H. Genest](https://orcid.org/0000-0002-4098-2024) [56],
[S. Gentile](https://orcid.org/0000-0003-4550-7174) [70a,70b] [, S. George](https://orcid.org/0000-0003-3565-3290) [91] [, W.F. George](https://orcid.org/0000-0003-3674-7475) [19] [, T. Geralis](https://orcid.org/0000-0001-7188-979X) [42], L.O. Gerlach [51] [, P. Gessinger-Befurt](https://orcid.org/0000-0002-3056-7417) [34],
[M. Ghasemi Bostanabad](https://orcid.org/0000-0003-3492-4538) [171] [, A. Ghosh](https://orcid.org/0000-0003-0819-1553) [166] [, A. Ghosh](https://orcid.org/0000-0002-5716-356X) [75] [, B. Giacobbe](https://orcid.org/0000-0003-2987-7642) [21b] [, S. Giagu](https://orcid.org/0000-0001-9192-3537) [70a,70b],
[N. Giangiacomi](https://orcid.org/0000-0001-7314-0168) [162] [, P. Giannetti](https://orcid.org/0000-0002-3721-9490) [69a] [, A. Giannini](https://orcid.org/0000-0002-5683-814X) [67a,67b] [, S.M. Gibson](https://orcid.org/0000-0002-1236-9249) [91] [, M. Gignac](https://orcid.org/0000-0003-4155-7844) [141] [, D.T. Gil](https://orcid.org/0000-0001-9021-8836) [81b],
[B.J. Gilbert](https://orcid.org/0000-0003-0731-710X) [37] [, D. Gillberg](https://orcid.org/0000-0003-0341-0171) [32] [, G. Gilles](https://orcid.org/0000-0001-8451-4604) [115] [, N.E.K. Gillwald](https://orcid.org/0000-0003-0848-329X) [44] [, D.M. Gingrich](https://orcid.org/0000-0002-2552-1449) [2,ai] [, M.P. Giordani](https://orcid.org/0000-0002-0792-6039) [64a,64c],
[S. Giovinazzo](https://orcid.org/0000-0003-0823-4760) [48] [, P.F. Giraud](https://orcid.org/0000-0002-8485-9351) [140] [, G. Giugliarelli](https://orcid.org/0000-0001-5765-1750) [64a,64c] [, D. Giugni](https://orcid.org/0000-0002-6976-0951) [66a] [, F. Giuli](https://orcid.org/0000-0002-8506-274X) [71a,71b] [, I. Gkialas](https://orcid.org/0000-0002-8402-723X) [8,h],
[P. Gkountoumis](https://orcid.org/0000-0003-2331-9922) [9] [, L.K. Gladilin](https://orcid.org/0000-0001-9422-8636) [109] [, C. Glasman](https://orcid.org/0000-0003-2025-3817) [95] [, G.R. Gledhill](https://orcid.org/0000-0001-7701-5030) [127], M. Glisic [127] [, I. Gnesi](https://orcid.org/0000-0002-0772-7312) [39b,d],
[M. Goblirsch-Kolb](https://orcid.org/0000-0002-2785-9654) [24], D. Godin [106] [, S. Goldfarb](https://orcid.org/0000-0002-1677-3097) [101] [, T. Golling](https://orcid.org/0000-0001-8535-6687) [52] [, D. Golubkov](https://orcid.org/0000-0002-5521-9793) [118] [, J.P. Gombas](https://orcid.org/0000-0002-8285-3570) [103],
[A. Gomes](https://orcid.org/0000-0002-5940-9893) [135a,135b] [, R. Goncalves Gama](https://orcid.org/0000-0002-8263-4263) [51] [, R. Gonc¸alo](https://orcid.org/0000-0002-3826-3442) [135a,135c] [, G. Gonella](https://orcid.org/0000-0002-0524-2477) [127] [, L. Gonella](https://orcid.org/0000-0002-4919-0808) [19],
[A. Gongadze](https://orcid.org/0000-0001-8183-1612) [77] [, F. Gonnella](https://orcid.org/0000-0003-0885-1654) [19] [, J.L. Gonski](https://orcid.org/0000-0003-2037-6315) [37] [, S. Gonz´alez de la Hoz](https://orcid.org/0000-0001-5304-5390) [169] [, S. Gonzalez Fernandez](https://orcid.org/0000-0001-8176-0201) [12],
[R. Gonzalez Lopez](https://orcid.org/0000-0003-2302-8754) [88] [, C. Gonzalez Renteria](https://orcid.org/0000-0003-0079-8924) [16] [, R. Gonzalez Suarez](https://orcid.org/0000-0002-6126-7230) [167] [, S. Gonzalez-Sevilla](https://orcid.org/0000-0003-4458-9403) [52],
[G.R. Gonzalvo Rodriguez](https://orcid.org/0000-0002-6816-4795) [169] [, R.Y. Gonz´alez Andana](https://orcid.org/0000-0002-0700-1757) [142a] [, L. Goossens](https://orcid.org/0000-0002-2536-4498) [34] [, N.A. Gorasia](https://orcid.org/0000-0002-7152-363X) [19],
[P.A. Gorbounov](https://orcid.org/0000-0001-9135-1516) [119] [, H.A. Gordon](https://orcid.org/0000-0003-4362-019X) [27] [, B. Gorini](https://orcid.org/0000-0003-4177-9666) [34] [, E. Gorini](https://orcid.org/0000-0002-7688-2797) [65a,65b] [, A. Goriˇsek](https://orcid.org/0000-0002-3903-3438) [89] [, A.T. Goshaw](https://orcid.org/0000-0002-5704-0885) [47],
[M.I. Gostkin](https://orcid.org/0000-0002-4311-3756) [77] [, C.A. Gottardo](https://orcid.org/0000-0003-0348-0364) [114] [, M. Gouighri](https://orcid.org/0000-0002-9551-0251) [33b] [, V. Goumarre](https://orcid.org/0000-0002-1294-9091) [44] [, A.G. Goussiou](https://orcid.org/0000-0001-6211-7122) [144] [, N. Govender](https://orcid.org/0000-0002-5068-5429) [31c],
[C. Goy](https://orcid.org/0000-0002-1297-8925) [4] [, I. Grabowska-Bold](https://orcid.org/0000-0001-9159-1210) [81a] [, K. Graham](https://orcid.org/0000-0002-5832-8653) [32] [, E. Gramstad](https://orcid.org/0000-0001-5792-5352) [129] [, S. Grancagnolo](https://orcid.org/0000-0001-8490-8304) [17] [, M. Grandi](https://orcid.org/0000-0002-5924-2544) [152],
V. Gratchev [133] [, P.M. Gravila](https://orcid.org/0000-0002-0154-577X) [25f] [, F.G. Gravili](https://orcid.org/0000-0003-2422-5960) [65a,65b] [, H.M. Gray](https://orcid.org/0000-0002-5293-4716) [16] [, C. Grefe](https://orcid.org/0000-0001-7050-5301) [22] [, I.M. Gregor](https://orcid.org/0000-0002-5976-7818) [44] [, P. Grenier](https://orcid.org/0000-0002-9926-5417) [149],
[K. Grevtsov](https://orcid.org/0000-0003-2704-6028) [44] [, C. Grieco](https://orcid.org/0000-0002-3955-4399) [12], N.A. Grieser [124], A.A. Grillo [141] [, K. Grimm](https://orcid.org/0000-0001-6587-7397) [29,l] [, S. Grinstein](https://orcid.org/0000-0002-6460-8694) [12,v] [, J.-F. Grivaz](https://orcid.org/0000-0003-4793-7995) [62],
[S. Groh](https://orcid.org/0000-0002-3001-3545) [96] [, E. Gross](https://orcid.org/0000-0003-1244-9350) [175] [, J. Grosse-Knetter](https://orcid.org/0000-0003-3085-7067) [51], C. Grud [102] [, A. Grummer](https://orcid.org/0000-0003-2752-1183) [113] [, J.C. Grundy](https://orcid.org/0000-0001-7136-0597) [130] [, L. Guan](https://orcid.org/0000-0003-1897-1617) [102],
[W. Guan](https://orcid.org/0000-0002-5548-5194) [176] [, C. Gubbels](https://orcid.org/0000-0003-2329-4219) [170] [, J. Guenther](https://orcid.org/0000-0003-3189-3959) [34] [, J.G.R. Guerrero Rojas](https://orcid.org/0000-0001-8487-3594) [169] [, F. Guescini](https://orcid.org/0000-0001-5351-2673) [111] [, D. Guest](https://orcid.org/0000-0002-4305-2295) [17],

[R. Gugel](https://orcid.org/0000-0002-3349-1163) [96] [, A. Guida](https://orcid.org/0000-0001-9021-9038) [44] [, T. Guillemin](https://orcid.org/0000-0001-9698-6000) [4] [, S. Guindon](https://orcid.org/0000-0001-7595-3859) [34] [, J. Guo](https://orcid.org/0000-0001-8125-9433) [58c] [, L. Guo](https://orcid.org/0000-0002-6785-9202) [62] [, Y. Guo](https://orcid.org/0000-0002-6027-5132) [102] [, R. Gupta](https://orcid.org/0000-0003-1510-3371) [44],
[S. Gurbuz](https://orcid.org/0000-0002-9152-1455) [22] [, G. Gustavino](https://orcid.org/0000-0002-5938-4921) [124] [, M. Guth](https://orcid.org/0000-0002-6647-1433) [52] [, P. Gutierrez](https://orcid.org/0000-0003-2326-3877) [124] [, L.F. Gutierrez Zagazeta](https://orcid.org/0000-0003-0374-1595) [132] [, C. Gutschow](https://orcid.org/0000-0003-0857-794X) [92],
[C. Guyot](https://orcid.org/0000-0002-2300-7497) [140] [, C. Gwenlan](https://orcid.org/0000-0002-3518-0617) [130] [, C.B. Gwilliam](https://orcid.org/0000-0002-9401-5304) [88] [, E.S. Haaland](https://orcid.org/0000-0002-3676-493X) [129] [, A. Haas](https://orcid.org/0000-0002-4832-0455) [121] [, M. Habedank](https://orcid.org/0000-0002-7412-9355) [44] [, C. Haber](https://orcid.org/0000-0002-0155-1360) [16],
[H.K. Hadavand](https://orcid.org/0000-0001-5447-3346) [7] [, A. Hadef](https://orcid.org/0000-0003-2508-0628) [96] [, S. Hadzic](https://orcid.org/0000-0002-8875-8523) [111] [, M. Haleem](https://orcid.org/0000-0003-3826-6333) [172] [, J. Haley](https://orcid.org/0000-0002-6938-7405) [125] [, J.J. Hall](https://orcid.org/0000-0002-8304-9170) [145] [, G. Halladjian](https://orcid.org/0000-0001-7162-0301) [103],
[G.D. Hallewell](https://orcid.org/0000-0001-6267-8560) [98] [, L. Halser](https://orcid.org/0000-0002-0759-7247) [18] [, K. Hamano](https://orcid.org/0000-0002-9438-8020) [171] [, H. Hamdaoui](https://orcid.org/0000-0001-5709-2100) [33e] [, M. Hamer](https://orcid.org/0000-0003-1550-2030) [22] [, G.N. Hamity](https://orcid.org/0000-0002-4537-0377) [48] [, K. Han](https://orcid.org/0000-0002-1627-4810) [58a],
[L. Han](https://orcid.org/0000-0003-3321-8412) [13c] [, L. Han](https://orcid.org/0000-0002-6353-9711) [58a] [, S. Han](https://orcid.org/0000-0001-8383-7348) [16] [, Y.F. Han](https://orcid.org/0000-0002-7084-8424) [162] [, K. Hanagaki](https://orcid.org/0000-0003-0676-0441) [79,t] [, M. Hance](https://orcid.org/0000-0001-8392-0934) [141] [, M.D. Hank](https://orcid.org/0000-0002-4731-6120) [35] [, R. Hankache](https://orcid.org/0000-0003-4519-8949) [97],
[E. Hansen](https://orcid.org/0000-0002-5019-1648) [94] [, J.B. Hansen](https://orcid.org/0000-0002-3684-8340) [38] [, J.D. Hansen](https://orcid.org/0000-0003-3102-0437) [38] [, M.C. Hansen](https://orcid.org/0000-0002-8892-4552) [22] [, P.H. Hansen](https://orcid.org/0000-0002-6764-4789) [38] [, K. Hara](https://orcid.org/0000-0003-1629-0535) [164] [, T. Harenberg](https://orcid.org/0000-0001-8682-3734) [177],
[S. Harkusha](https://orcid.org/0000-0002-0309-4490) [104] [, Y.T. Harris](https://orcid.org/0000-0001-5816-2158) [130], P.F. Harrison [173] [, N.M. Hartman](https://orcid.org/0000-0001-9111-4916) [149] [, N.M. Hartmann](https://orcid.org/0000-0003-0047-2908) [110] [, Y. Hasegawa](https://orcid.org/0000-0003-2683-7389) [146],
[A. Hasib](https://orcid.org/0000-0003-0457-2244) [48] [, S. Hassani](https://orcid.org/0000-0002-2834-5110) [140] [, S. Haug](https://orcid.org/0000-0003-0442-3361) [18] [, R. Hauser](https://orcid.org/0000-0001-7682-8857) [103] [, M. Havranek](https://orcid.org/0000-0002-3031-3222) [137] [, C.M. Hawkes](https://orcid.org/0000-0001-9167-0592) [19] [, R.J. Hawkings](https://orcid.org/0000-0001-9719-0290) [34],
[S. Hayashida](https://orcid.org/0000-0002-5924-3803) [112] [, D. Hayden](https://orcid.org/0000-0001-5220-2972) [103] [, C. Hayes](https://orcid.org/0000-0002-0298-0351) [102] [, R.L. Hayes](https://orcid.org/0000-0001-7752-9285) [170] [, C.P. Hays](https://orcid.org/0000-0003-2371-9723) [130] [, J.M. Hays](https://orcid.org/0000-0003-1554-5401) [90] [, H.S. Hayward](https://orcid.org/0000-0002-0972-3411) [88],
[S.J. Haywood](https://orcid.org/0000-0003-2074-013X) [139] [, F. He](https://orcid.org/0000-0003-3733-4058) [58a] [, Y. He](https://orcid.org/0000-0002-0619-1579) [160] [, Y. He](https://orcid.org/0000-0001-8068-5596) [131] [, M.P. Heath](https://orcid.org/0000-0003-2945-8448) [48] [, V. Hedberg](https://orcid.org/0000-0002-4596-3965) [94] [, A.L. Heggelund](https://orcid.org/0000-0002-7736-2806) [129],
[N.D. Hehir](https://orcid.org/0000-0003-0466-4472) [90] [, C. Heidegger](https://orcid.org/0000-0001-8821-1205) [50] [, K.K. Heidegger](https://orcid.org/0000-0003-3113-0484) [50] [, W.D. Heidorn](https://orcid.org/0000-0001-9539-6957) [76] [, J. Heilman](https://orcid.org/0000-0001-6792-2294) [32] [, S. Heim](https://orcid.org/0000-0002-2639-6571) [44] [, T. Heim](https://orcid.org/0000-0002-7669-5318) [16],
[B. Heinemann](https://orcid.org/0000-0002-1673-7926) [44,ag] [, J.G. Heinlein](https://orcid.org/0000-0001-6878-9405) [132] [, J.J. Heinrich](https://orcid.org/0000-0002-0253-0924) [127] [, L. Heinrich](https://orcid.org/0000-0002-4048-7584) [34] [, J. Hejbal](https://orcid.org/0000-0002-4600-3659) [136] [, L. Helary](https://orcid.org/0000-0001-7891-8354) [44] [, A. Held](https://orcid.org/0000-0002-8924-5885) [121],
[C.M. Helling](https://orcid.org/0000-0002-2657-7532) [141] [, S. Hellman](https://orcid.org/0000-0002-5415-1600) [43a,43b] [, C. Helsens](https://orcid.org/0000-0002-9243-7554) [34], R.C.W. Henderson [87] [, L. Henkelmann](https://orcid.org/0000-0001-8231-2080) [30],
A.M. Henriques Correia [34] [, H. Herde](https://orcid.org/0000-0001-8926-6734) [149] [, Y. Hern´andez Jim´enez](https://orcid.org/0000-0001-9844-6200) [151], H. Herr [96] [, M.G. Herrmann](https://orcid.org/0000-0002-2254-0257) [110],
[T. Herrmann](https://orcid.org/0000-0002-1478-3152) [46] [, G. Herten](https://orcid.org/0000-0001-7661-5122) [50] [, R. Hertenberger](https://orcid.org/0000-0002-2646-5805) [110] [, L. Hervas](https://orcid.org/0000-0002-0778-2717) [34] [, N.P. Hessey](https://orcid.org/0000-0002-6698-9937) [163a] [, H. Hibi](https://orcid.org/0000-0002-4630-9914) [80] [, S. Higashino](https://orcid.org/0000-0002-5704-4253) [79],
[E. Hig´on-Rodriguez](https://orcid.org/0000-0002-3094-2520) [169], K.H. Hiller [44] [, S.J. Hillier](https://orcid.org/0000-0002-7599-6469) [19] [, M. Hils](https://orcid.org/0000-0002-8616-5898) [46] [, I. Hinchliffe](https://orcid.org/0000-0002-5529-2173) [16] [, F. Hinterkeuser](https://orcid.org/0000-0002-0556-189X) [22],
[M. Hirose](https://orcid.org/0000-0003-4988-9149) [128] [, S. Hirose](https://orcid.org/0000-0002-2389-1286) [164] [, D. Hirschbuehl](https://orcid.org/0000-0002-7998-8925) [177] [, B. Hiti](https://orcid.org/0000-0002-8668-6933) [89], O. Hladik [136] [, J. Hobbs](https://orcid.org/0000-0001-5404-7857) [151] [, R. Hobincu](https://orcid.org/0000-0001-7602-5771) [25e],
[N. Hod](https://orcid.org/0000-0001-5241-0544) [175] [, M.C. Hodgkinson](https://orcid.org/0000-0002-1040-1241) [145] [, B.H. Hodkinson](https://orcid.org/0000-0002-2244-189X) [30] [, A. Hoecker](https://orcid.org/0000-0002-6596-9395) [34] [, J. Hofer](https://orcid.org/0000-0003-2799-5020) [44] [, D. Hohn](https://orcid.org/0000-0002-5317-1247) [50] [, T. Holm](https://orcid.org/0000-0001-5407-7247) [22],
[T.R. Holmes](https://orcid.org/0000-0002-3959-5174) [35] [, M. Holzbock](https://orcid.org/0000-0001-8018-4185) [111] [, L.B.A.H. Hommels](https://orcid.org/0000-0003-0684-600X) [30] [, B.P. Honan](https://orcid.org/0000-0002-2698-4787) [97] [, J. Hong](https://orcid.org/0000-0002-7494-5504) [58c] [, T.M. Hong](https://orcid.org/0000-0001-7834-328X) [134],
[Y. Hong](https://orcid.org/0000-0003-4752-2458) [51] [, J.C. Honig](https://orcid.org/0000-0002-3596-6572) [50] [, A. H¨onle](https://orcid.org/0000-0001-6063-2884) [111] [, B.H. Hooberman](https://orcid.org/0000-0002-4090-6099) [168] [, W.H. Hopkins](https://orcid.org/0000-0001-7814-8740) [5] [, Y. Horii](https://orcid.org/0000-0003-0457-3052) [112] [, L.A. Horyn](https://orcid.org/0000-0002-9512-4932) [35],
[S. Hou](https://orcid.org/0000-0001-9861-151X) [154] [, J. Howarth](https://orcid.org/0000-0002-0560-8985) [55] [, J. Hoya](https://orcid.org/0000-0002-7562-0234) [86] [, M. Hrabovsky](https://orcid.org/0000-0003-4223-7316) [126] [, A. Hrynevich](https://orcid.org/0000-0002-5411-114X) [105] [, T. Hryn’ova](https://orcid.org/0000-0001-5914-8614) [4] [, P.J. Hsu](https://orcid.org/0000-0003-3895-8356) [61],
[S.-C. Hsu](https://orcid.org/0000-0001-6214-8500) [144] [, Q. Hu](https://orcid.org/0000-0002-9705-7518) [37] [, S. Hu](https://orcid.org/0000-0003-4696-4430) [58c] [, Y.F. Hu](https://orcid.org/0000-0002-0552-3383) [13a,13d,ak] [, D.P. Huang](https://orcid.org/0000-0002-1753-5621) [92] [, X. Huang](https://orcid.org/0000-0002-6617-3807) [13c] [, Y. Huang](https://orcid.org/0000-0003-1826-2749) [58a] [, Y. Huang](https://orcid.org/0000-0002-5972-2855) [13a],
[Z. Hubacek](https://orcid.org/0000-0003-3250-9066) [137] [, F. Hubaut](https://orcid.org/0000-0002-0113-2465) [98] [, M. Huebner](https://orcid.org/0000-0002-1162-8763) [22] [, F. Huegging](https://orcid.org/0000-0002-7472-3151) [22] [, T.B. Huffman](https://orcid.org/0000-0002-5332-2738) [130] [, M. Huhtinen](https://orcid.org/0000-0002-1752-3583) [34],
[S.K. Huiberts](https://orcid.org/0000-0002-3277-7418) [15] [, R. Hulsken](https://orcid.org/0000-0002-0095-1290) [56] [, N. Huseynov](https://orcid.org/0000-0003-2201-5572) [77,z] [, J. Huston](https://orcid.org/0000-0001-9097-3014) [103] [, J. Huth](https://orcid.org/0000-0002-6867-2538) [57] [, R. Hyneman](https://orcid.org/0000-0002-9093-7141) [149] [, S. Hyrych](https://orcid.org/0000-0001-9425-4287) [26a],
[G. Iacobucci](https://orcid.org/0000-0001-9965-5442) [52] [, G. Iakovidis](https://orcid.org/0000-0002-0330-5921) [27] [, I. Ibragimov](https://orcid.org/0000-0001-8847-7337) [147] [, L. Iconomidou-Fayard](https://orcid.org/0000-0001-6334-6648) [62] [, P. Iengo](https://orcid.org/0000-0002-5035-1242) [34] [, R. Iguchi](https://orcid.org/0000-0002-0940-244X) [159],
[T. Iizawa](https://orcid.org/0000-0001-5312-4865) [52] [, Y. Ikegami](https://orcid.org/0000-0001-7287-6579) [79] [, A. Ilg](https://orcid.org/0000-0001-9488-8095) [18] [, N. Ilic](https://orcid.org/0000-0003-0105-7634) [162] [, H. Imam](https://orcid.org/0000-0002-7854-3174) [33a] [, T. Ingebretsen Carlson](https://orcid.org/0000-0002-3699-8517) [43a,43b] [, G. Introzzi](https://orcid.org/0000-0002-1314-2580) [68a,68b],


62


[M. Iodice](https://orcid.org/0000-0003-4446-8150) [72a] [, V. Ippolito](https://orcid.org/0000-0001-5126-1620) [70a,70b] [, M. Ishino](https://orcid.org/0000-0002-7185-1334) [159] [, W. Islam](https://orcid.org/0000-0002-5624-5934) [176] [, C. Issever](https://orcid.org/0000-0001-8259-1067) [17,44] [, S. Istin](https://orcid.org/0000-0001-8504-6291) [11c,al],
[J.M. Iturbe Ponce](https://orcid.org/0000-0002-2325-3225) [60a] [, R. Iuppa](https://orcid.org/0000-0001-5038-2762) [73a,73b] [, A. Ivina](https://orcid.org/0000-0002-9152-383X) [175] [, J.M. Izen](https://orcid.org/0000-0002-9846-5601) [41] [, V. Izzo](https://orcid.org/0000-0002-8770-1592) [67a] [, P. Jacka](https://orcid.org/0000-0003-2489-9930) [136] [, P. Jackson](https://orcid.org/0000-0002-0847-402X) [1],
[R.M. Jacobs](https://orcid.org/0000-0001-5446-5901) [44] [, B.P. Jaeger](https://orcid.org/0000-0002-5094-5067) [148] [, C.S. Jagfeld](https://orcid.org/0000-0002-1669-759X) [110] [, G. J¨akel](https://orcid.org/0000-0001-5687-1006) [177] [, K. Jakobs](https://orcid.org/0000-0001-8885-012X) [50] [, T. Jakoubek](https://orcid.org/0000-0001-7038-0369) [175] [, J. Jamieson](https://orcid.org/0000-0001-9554-0787) [55],
[K.W. Janas](https://orcid.org/0000-0001-5411-8934) [81a] [, G. Jarlskog](https://orcid.org/0000-0002-8731-2060) [94] [, A.E. Jaspan](https://orcid.org/0000-0003-4189-2837) [88], N. Javadov [77,z] [, T. Jav˚urek](https://orcid.org/0000-0002-9389-3682) [34] [, M. Javurkova](https://orcid.org/0000-0001-8798-808X) [99] [, F. Jeanneau](https://orcid.org/0000-0002-6360-6136) [140],
[L. Jeanty](https://orcid.org/0000-0001-6507-4623) [127] [, J. Jejelava](https://orcid.org/0000-0002-0159-6593) [155a,aa] [, P. Jenni](https://orcid.org/0000-0002-4539-4192) [50,e] [, S. J´ez´equel](https://orcid.org/0000-0001-7369-6975) [4] [, J. Jia](https://orcid.org/0000-0002-5725-3397) [151] [, Z. Jia](https://orcid.org/0000-0002-2657-3099) [13c], Y. Jiang [58a] [, S. Jiggins](https://orcid.org/0000-0003-2906-1977) [48],
[J. Jimenez Pena](https://orcid.org/0000-0002-8705-628X) [111] [, S. Jin](https://orcid.org/0000-0002-5076-7803) [13c] [, A. Jinaru](https://orcid.org/0000-0001-7449-9164) [25b] [, O. Jinnouchi](https://orcid.org/0000-0001-5073-0974) [160] [, H. Jivan](https://orcid.org/0000-0002-4115-6322) [31f] [, P. Johansson](https://orcid.org/0000-0001-5410-1315) [145] [, K.A. Johns](https://orcid.org/0000-0001-9147-6052) [6],
[C.A. Johnson](https://orcid.org/0000-0002-5387-572X) [63] [, D.M. Jones](https://orcid.org/0000-0002-9204-4689) [30] [, E. Jones](https://orcid.org/0000-0001-6289-2292) [173] [, R.W.L. Jones](https://orcid.org/0000-0002-6427-3513) [87] [, T.J. Jones](https://orcid.org/0000-0002-2580-1977) [88] [, J. Jovicevic](https://orcid.org/0000-0001-5650-4556) [14] [, X. Ju](https://orcid.org/0000-0002-9745-1638) [16],
[J.J. Junggeburth](https://orcid.org/0000-0001-7205-1171) [34] [, A. Juste Rozas](https://orcid.org/0000-0002-1558-3291) [12,v] [, S. Kabana](https://orcid.org/0000-0003-0568-5750) [142d] [, A. Kaczmarska](https://orcid.org/0000-0002-8880-4120) [82], M. Kado [70a,70b] [, H. Kagan](https://orcid.org/0000-0002-4693-7857) [123],
[M. Kagan](https://orcid.org/0000-0002-3386-6869) [149], A. Kahn [37] [, A. Kahn](https://orcid.org/0000-0001-7131-3029) [132] [, C. Kahra](https://orcid.org/0000-0002-9003-5711) [96] [, T. Kaji](https://orcid.org/0000-0002-6532-7501) [174] [, E. Kajomovitz](https://orcid.org/0000-0002-8464-1790) [156] [, C.W. Kalderon](https://orcid.org/0000-0002-2875-853X) [27],
[A. Kamenshchikov](https://orcid.org/0000-0002-7845-2301) [118] [, M. Kaneda](https://orcid.org/0000-0003-1510-7719) [159] [, N.J. Kang](https://orcid.org/0000-0001-5009-0399) [141] [, S. Kang](https://orcid.org/0000-0002-5320-7043) [76] [, Y. Kano](https://orcid.org/0000-0003-1090-3820) [112] [, D. Kar](https://orcid.org/0000-0002-4238-9822) [31f] [, K. Karava](https://orcid.org/0000-0002-5010-8613) [130],
[M.J. Kareem](https://orcid.org/0000-0001-8967-1705) [163b] [, I. Karkanias](https://orcid.org/0000-0002-6940-261X) [158] [, S.N. Karpov](https://orcid.org/0000-0002-2230-5353) [77] [, Z.M. Karpova](https://orcid.org/0000-0003-0254-4629) [77] [, V. Kartvelishvili](https://orcid.org/0000-0002-1957-3787) [87] [, A.N. Karyukhin](https://orcid.org/0000-0001-9087-4315) [118],
[E. Kasimi](https://orcid.org/0000-0002-7139-8197) [158] [, C. Kato](https://orcid.org/0000-0002-0794-4325) [58d] [, J. Katzy](https://orcid.org/0000-0003-3121-395X) [44] [, K. Kawade](https://orcid.org/0000-0002-7874-6107) [146] [, K. Kawagoe](https://orcid.org/0000-0001-8882-129X) [85] [, T. Kawaguchi](https://orcid.org/0000-0002-9124-788X) [112] [, T. Kawamoto](https://orcid.org/0000-0002-5841-5511) [140],
G. Kawamura [51] [, E.F. Kay](https://orcid.org/0000-0002-6304-3230) [171] [, F.I. Kaya](https://orcid.org/0000-0002-9775-7303) [165] [, S. Kazakos](https://orcid.org/0000-0002-7252-3201) [12] [, V.F. Kazanin](https://orcid.org/0000-0002-4906-5468) [117b,117a] [, Y. Ke](https://orcid.org/0000-0001-5798-6665) [151],
[J.M. Keaveney](https://orcid.org/0000-0003-0766-5307) [31a] [, R. Keeler](https://orcid.org/0000-0002-0510-4189) [171] [, J.S. Keller](https://orcid.org/0000-0001-7140-9813) [32], A.S. Kelly [92] [, D. Kelsey](https://orcid.org/0000-0002-2297-1356) [152] [, J.J. Kempster](https://orcid.org/0000-0003-4168-3373) [19] [, J. Kendrick](https://orcid.org/0000-0001-9845-5473) [19],
[K.E. Kennedy](https://orcid.org/0000-0003-3264-548X) [37] [, O. Kepka](https://orcid.org/0000-0002-2555-497X) [136] [, S. Kersten](https://orcid.org/0000-0002-0511-2592) [177] [, B.P. Kersevan](https://orcid.org/0000-0002-4529-452X) ˇ [89] [, S. Ketabchi Haghighat](https://orcid.org/0000-0002-8597-3834) [162] [, M. Khandoga](https://orcid.org/0000-0002-8785-7378) [131],
[A. Khanov](https://orcid.org/0000-0001-9621-422X) [125] [, A.G. Kharlamov](https://orcid.org/0000-0002-1051-3833) [117b,117a] [, T. Kharlamova](https://orcid.org/0000-0002-0387-6804) [117b,117a] [, E.E. Khoda](https://orcid.org/0000-0001-8720-6615) [144] [, T.J. Khoo](https://orcid.org/0000-0002-5954-3101) [17],
[G. Khoriauli](https://orcid.org/0000-0002-6353-8452) [172] [, E. Khramov](https://orcid.org/0000-0001-7400-6454) [77] [, J. Khubua](https://orcid.org/0000-0003-2350-1249) [155b] [, S. Kido](https://orcid.org/0000-0003-0536-5386) [80] [, M. Kiehn](https://orcid.org/0000-0001-9608-2626) [34] [, A. Kilgallon](https://orcid.org/0000-0003-1450-0009) [127] [, E. Kim](https://orcid.org/0000-0002-4203-014X) [160],
[Y.K. Kim](https://orcid.org/0000-0003-3286-1326) [35] [, N. Kimura](https://orcid.org/0000-0002-8883-9374) [92], E. Kioseoglou [48] [, A. Kirchhoff](https://orcid.org/0000-0001-5611-9543) [51] [, D. Kirchmeier](https://orcid.org/0000-0001-8545-5650) [46] [, C. Kirfel](https://orcid.org/0000-0003-1679-6907) [22] [, J. Kirk](https://orcid.org/0000-0001-8096-7577) [139],
[A.E. Kiryunin](https://orcid.org/0000-0001-7490-6890) [111] [, T. Kishimoto](https://orcid.org/0000-0003-3476-8192) [159], D.P. Kisliuk [162] [, C. Kitsaki](https://orcid.org/0000-0003-4431-8400) [9] [, O. Kivernyk](https://orcid.org/0000-0002-6854-2717) [22],
[T. Klapdor-Kleingrothaus](https://orcid.org/0000-0003-1423-6041) [50] [, M. Klassen](https://orcid.org/0000-0002-4326-9742) [59a] [, C. Klein](https://orcid.org/0000-0002-3780-1755) [32] [, L. Klein](https://orcid.org/0000-0002-0145-4747) [172] [, M.H. Klein](https://orcid.org/0000-0002-9999-2534) [102] [, M. Klein](https://orcid.org/0000-0002-8527-964X) [88] [, U. Klein](https://orcid.org/0000-0001-7391-5330) [88],
[P. Klimek](https://orcid.org/0000-0003-1661-6873) [34] [, A. Klimentov](https://orcid.org/0000-0003-2748-4829) [27] [, F. Klimpel](https://orcid.org/0000-0002-9362-3973) [111] [, T. Klingl](https://orcid.org/0000-0002-5721-9834) [22] [, T. Klioutchnikova](https://orcid.org/0000-0002-9580-0363) [34] [, F.F. Klitzner](https://orcid.org/0000-0002-7864-459X) [110] [, P. Kluit](https://orcid.org/0000-0001-6419-5829) [115],
[S. Kluth](https://orcid.org/0000-0001-8484-2261) [111] [, E. Kneringer](https://orcid.org/0000-0002-6206-1912) [74] [, T.M. Knight](https://orcid.org/0000-0003-2486-7672) [162] [, A. Knue](https://orcid.org/0000-0002-1559-9285) [50], D. Kobayashi [85] [, R. Kobayashi](https://orcid.org/0000-0002-7584-078X) [83] [, M. Kobel](https://orcid.org/0000-0002-0124-2699) [46],
[M. Kocian](https://orcid.org/0000-0003-4559-6058) [149], T. Kodama [159] [, P. Kodys](https://orcid.org/0000-0002-8644-2349) [138] [, D.M. Koeck](https://orcid.org/0000-0002-9090-5502) [152] [, P.T. Koenig](https://orcid.org/0000-0002-0497-3550) [22] [, T. Koffas](https://orcid.org/0000-0001-9612-4988) [32] [, N.M. K¨ohler](https://orcid.org/0000-0002-0490-9778) [34],
[M. Kolb](https://orcid.org/0000-0002-6117-3816) [140] [, I. Koletsou](https://orcid.org/0000-0002-8560-8917) [4] [, T. Komarek](https://orcid.org/0000-0002-3047-3146) [126] [, K. K¨oneke](https://orcid.org/0000-0002-6901-9717) [50] [, A.X.Y. Kong](https://orcid.org/0000-0001-8063-8765) [1] [, T. Kono](https://orcid.org/0000-0003-1553-2950) [122], V. Konstantinides [92],
[N. Konstantinidis](https://orcid.org/0000-0002-4140-6360) [92] [, B. Konya](https://orcid.org/0000-0002-1859-6557) [94] [, R. Kopeliansky](https://orcid.org/0000-0002-8775-1194) [63] [, S. Koperny](https://orcid.org/0000-0002-2023-5945) [81a] [, K. Korcyl](https://orcid.org/0000-0001-8085-4505) [82] [, K. Kordas](https://orcid.org/0000-0003-0486-2081) [158],
G. Koren [157] [, A. Korn](https://orcid.org/0000-0002-3962-2099) [92] [, S. Korn](https://orcid.org/0000-0001-9291-5408) [51] [, I. Korolkov](https://orcid.org/0000-0002-9211-9775) [12], E.V. Korolkova [145] [, N. Korotkova](https://orcid.org/0000-0003-3640-8676) [109] [, B. Kortman](https://orcid.org/0000-0001-7081-3275) [115],
[O. Kortner](https://orcid.org/0000-0003-0352-3096) [111] [, S. Kortner](https://orcid.org/0000-0001-8667-1814) [111] [, W.H. Kostecka](https://orcid.org/0000-0003-1772-6898) [116] [, V.V. Kostyukhin](https://orcid.org/0000-0002-0490-9209) [147,161] [, A. Kotsokechagia](https://orcid.org/0000-0002-8057-9467) [62] [, A. Kotwal](https://orcid.org/0000-0003-3384-5053) [47],
[A. Koulouris](https://orcid.org/0000-0003-1012-4675) [34] [, A. Kourkoumeli-Charalampidi](https://orcid.org/0000-0002-6614-108X) [68a,68b] [, C. Kourkoumelis](https://orcid.org/0000-0003-0083-274X) [8] [, E. Kourlitis](https://orcid.org/0000-0001-6568-2047) [5] [, O. Kovanda](https://orcid.org/0000-0003-0294-3953) [152],
[R. Kowalewski](https://orcid.org/0000-0002-7314-0990) [171] [, W. Kozanecki](https://orcid.org/0000-0001-6226-8385) [140] [, A.S. Kozhin](https://orcid.org/0000-0003-4724-9017) [118] [, V.A. Kramarenko](https://orcid.org/0000-0002-8625-5586) [109] [, G. Kramberger](https://orcid.org/0000-0002-7580-384X) [89] [, P. Kramer](https://orcid.org/0000-0002-0296-5899) [96],
[D. Krasnopevtsev](https://orcid.org/0000-0002-6356-372X) [58a] [, M.W. Krasny](https://orcid.org/0000-0002-7440-0520) [131] [, A. Krasznahorkay](https://orcid.org/0000-0002-6468-1381) [34] [, J.A. Kremer](https://orcid.org/0000-0003-4487-6365) [96] [, J. Kretzschmar](https://orcid.org/0000-0002-8515-1355) [88] [, K. Kreul](https://orcid.org/0000-0002-1739-6596) [17],
[P. Krieger](https://orcid.org/0000-0001-9958-949X) [162] [, F. Krieter](https://orcid.org/0000-0002-7675-8024) [110] [, S. Krishnamurthy](https://orcid.org/0000-0001-6169-0517) [99] [, A. Krishnan](https://orcid.org/0000-0002-0734-6122) [59b] [, M. Krivos](https://orcid.org/0000-0001-9062-2257) [138] [, K. Krizka](https://orcid.org/0000-0001-6408-2648) [16],
[K. Kroeninger](https://orcid.org/0000-0001-9873-0228) [45] [, H. Kroha](https://orcid.org/0000-0003-1808-0259) [111] [, J. Kroll](https://orcid.org/0000-0001-6215-3326) [136] [, J. Kroll](https://orcid.org/0000-0002-0964-6815) [132] [, K.S. Krowpman](https://orcid.org/0000-0001-9395-3430) [103] [, U. Kruchonak](https://orcid.org/0000-0003-2116-4592) [77] [, H. Kr¨uger](https://orcid.org/0000-0001-8287-3961) [22],
N. Krumnack [76] [, M.C. Kruse](https://orcid.org/0000-0001-5791-0345) [47] [, J.A. Krzysiak](https://orcid.org/0000-0002-1214-9262) [82] [, A. Kubota](https://orcid.org/0000-0003-3993-4903) [160] [, O. Kuchinskaia](https://orcid.org/0000-0002-3664-2465) [161] [, S. Kuday](https://orcid.org/0000-0002-0116-5494) [3a],
[D. Kuechler](https://orcid.org/0000-0003-4087-1575) [44] [, J.T. Kuechler](https://orcid.org/0000-0001-9087-6230) [44] [, S. Kuehn](https://orcid.org/0000-0001-5270-0920) [34] [, T. Kuhl](https://orcid.org/0000-0002-1473-350X) [44] [, V. Kukhtin](https://orcid.org/0000-0003-4387-8756) [77] [, Y. Kulchitsky](https://orcid.org/0000-0002-3036-5575) [104,ad] [, S. Kuleshov](https://orcid.org/0000-0002-3065-326X) [142c],
[M. Kumar](https://orcid.org/0000-0003-3681-1588) [31f] [, N. Kumari](https://orcid.org/0000-0001-9174-6200) [98] [, M. Kuna](https://orcid.org/0000-0002-3598-2847) [56] [, A. Kupco](https://orcid.org/0000-0003-3692-1410) [136], T. Kupfer [45] [, O. Kuprash](https://orcid.org/0000-0002-7540-0012) [50] [, H. Kurashige](https://orcid.org/0000-0003-3932-016X) [80],
[L.L. Kurchaninov](https://orcid.org/0000-0001-9392-3936) [163a] [, Y.A. Kurochkin](https://orcid.org/0000-0002-1281-8462) [104] [, A. Kurova](https://orcid.org/0000-0001-7924-1517) [108], M.G. Kurth [13a,13d] [, E.S. Kuwertz](https://orcid.org/0000-0002-1921-6173) [34] [, M. Kuze](https://orcid.org/0000-0001-8858-8440) [160],
[A.K. Kvam](https://orcid.org/0000-0001-7243-0227) [144] [, J. Kvita](https://orcid.org/0000-0001-5973-8729) [126] [, T. Kwan](https://orcid.org/0000-0001-8717-4449) [100] [, K.W. Kwok](https://orcid.org/0000-0002-0820-9998) [60a] [, C. Lacasta](https://orcid.org/0000-0002-2623-6252) [169] [, F. Lacava](https://orcid.org/0000-0003-4588-8325) [70a,70b] [, H. Lacker](https://orcid.org/0000-0002-7183-8607) [17],
[D. Lacour](https://orcid.org/0000-0002-1590-194X) [131] [, N.N. Lad](https://orcid.org/0000-0002-3707-9010) [92] [, E. Ladygin](https://orcid.org/0000-0001-6206-8148) [77] [, R. Lafaye](https://orcid.org/0000-0001-7848-6088) [4] [, B. Laforge](https://orcid.org/0000-0002-4209-4194) [131] [, T. Lagouri](https://orcid.org/0000-0001-7509-7765) [142d] [, S. Lai](https://orcid.org/0000-0002-9898-9253) [51],
[I.K. Lakomiec](https://orcid.org/0000-0002-4357-7649) [81a] [, N. Lalloue](https://orcid.org/0000-0003-0953-559X) [56] [, J.E. Lambert](https://orcid.org/0000-0002-5606-4164) [124], S. Lammers [63] [, W. Lampl](https://orcid.org/0000-0002-2337-0958) [6] [, C. Lampoudis](https://orcid.org/0000-0001-9782-9920) [158],
[E. Lanc¸on](https://orcid.org/0000-0002-0225-187X) [27] [, U. Landgraf](https://orcid.org/0000-0002-8222-2066) [50] [, M.P.J. Landon](https://orcid.org/0000-0001-6828-9769) [90] [, V.S. Lang](https://orcid.org/0000-0001-9954-7898) [50] [, J.C. Lange](https://orcid.org/0000-0003-1307-1441) [51] [, R.J. Langenberg](https://orcid.org/0000-0001-6595-1382) [99],
[A.J. Lankford](https://orcid.org/0000-0001-8057-4351) [166] [, F. Lanni](https://orcid.org/0000-0002-7197-9645) [27] [, K. Lantzsch](https://orcid.org/0000-0002-0729-6487) [22] [, A. Lanza](https://orcid.org/0000-0003-4980-6032) [68a] [, A. Lapertosa](https://orcid.org/0000-0001-6246-6787) [53b,53a] [, J.F. Laporte](https://orcid.org/0000-0002-4815-5314) [140] [, T. Lari](https://orcid.org/0000-0002-1388-869X) [66a],
[F. Lasagni Manghi](https://orcid.org/0000-0001-6068-4473) [21b] [, M. Lassnig](https://orcid.org/0000-0002-9541-0592) [34] [, V. Latonova](https://orcid.org/0000-0001-9591-5622) [136] [, T.S. Lau](https://orcid.org/0000-0001-7110-7823) [60a] [, A. Laudrain](https://orcid.org/0000-0001-6098-0555) [96] [, A. Laurier](https://orcid.org/0000-0002-2575-0743) [32],
[M. Lavorgna](https://orcid.org/0000-0002-3407-752X) [67a,67b] [, S.D. Lawlor](https://orcid.org/0000-0003-3211-067X) [91] [, Z. Lawrence](https://orcid.org/0000-0002-9035-9679) [97] [, M. Lazzaroni](https://orcid.org/0000-0002-4094-1273) [66a,66b], B. Le [97] [, B. Leban](https://orcid.org/0000-0003-1501-7262) [89],
[A. Lebedev](https://orcid.org/0000-0002-9566-1850) [76] [, M. LeBlanc](https://orcid.org/0000-0001-5977-6418) [34] [, T. LeCompte](https://orcid.org/0000-0002-9450-6568) [5] [, F. Ledroit-Guillon](https://orcid.org/0000-0001-9398-1909) [56], A.C.A. Lee [92] [, G.R. Lee](https://orcid.org/0000-0002-5968-6954) [15] [, L. Lee](https://orcid.org/0000-0002-5590-335X) [57],
[S.C. Lee](https://orcid.org/0000-0002-3353-2658) [154] [, S. Lee](https://orcid.org/0000-0001-5688-1212) [76] [, L.L. Leeuw](https://orcid.org/0000-0002-3365-6781) [31c] [, B. Lefebvre](https://orcid.org/0000-0001-8212-6624) [163a] [, H.P. Lefebvre](https://orcid.org/0000-0002-7394-2408) [91] [, M. Lefebvre](https://orcid.org/0000-0002-5560-0586) [171] [, C. Leggett](https://orcid.org/0000-0002-9299-9020) [16],
[K. Lehmann](https://orcid.org/0000-0002-8590-8231) [148] [, N. Lehmann](https://orcid.org/0000-0001-5521-1655) [18] [, G. Lehmann Miotto](https://orcid.org/0000-0001-9045-7853) [34] [, W.A. Leight](https://orcid.org/0000-0002-2968-7841) [44] [, A. Leisos](https://orcid.org/0000-0002-8126-3958) [158,u] [, M.A.L. Leite](https://orcid.org/0000-0003-0392-3663) [78c],
[C.E. Leitgeb](https://orcid.org/0000-0002-0335-503X) [44] [, R. Leitner](https://orcid.org/0000-0002-2994-2187) [138] [, K.J.C. Leney](https://orcid.org/0000-0002-1525-2695) [40] [, T. Lenz](https://orcid.org/0000-0002-9560-1778) [22] [, S. Leone](https://orcid.org/0000-0001-6222-9642) [69a] [, C. Leonidopoulos](https://orcid.org/0000-0002-7241-2114) [48] [, A. Leopold](https://orcid.org/0000-0001-9415-7903) [150],


63


[C. Leroy](https://orcid.org/0000-0003-3105-7045) [106] [, R. Les](https://orcid.org/0000-0002-8875-1399) [103] [, C.G. Lester](https://orcid.org/0000-0001-5770-4883) [30] [, M. Levchenko](https://orcid.org/0000-0002-5495-0656) [133] [, J. Levˆeque](https://orcid.org/0000-0002-0244-4743) [4] [, D. Levin](https://orcid.org/0000-0003-0512-0856) [102] [, L.J. Levinson](https://orcid.org/0000-0003-4679-0485) [175],
[D.J. Lewis](https://orcid.org/0000-0002-7814-8596) [19] [, B. Li](https://orcid.org/0000-0002-7004-3802) [13b] [, B. Li](https://orcid.org/0000-0002-1974-2229) [58b], C. Li [58a] [, C-Q. Li](https://orcid.org/0000-0003-3495-7778) [58c,58d] [, H. Li](https://orcid.org/0000-0002-1081-2032) [58a] [, H. Li](https://orcid.org/0000-0002-4732-5633) [58b] [, H. Li](https://orcid.org/0000-0001-9346-6982) [58b] [, J. Li](https://orcid.org/0000-0003-4776-4123) [58c] [, K. Li](https://orcid.org/0000-0002-2545-0329) [144],
[L. Li](https://orcid.org/0000-0001-6411-6107) [58c] [, M. Li](https://orcid.org/0000-0003-4317-3203) [13a,13d] [, Q.Y. Li](https://orcid.org/0000-0001-6066-195X) [58a] [, S. Li](https://orcid.org/0000-0001-7879-3272) [58d,58c,c] [, T. Li](https://orcid.org/0000-0001-7775-4300) [58b] [, X. Li](https://orcid.org/0000-0001-6975-102X) [44] [, Y. Li](https://orcid.org/0000-0003-3042-0893) [44] [, Z. Li](https://orcid.org/0000-0003-1189-3505) [58b] [, Z. Li](https://orcid.org/0000-0001-9800-2626) [130] [, Z. Li](https://orcid.org/0000-0001-7096-2158) [100],
Z. Li [88] [, Z. Liang](https://orcid.org/0000-0003-0629-2131) [13a] [, M. Liberatore](https://orcid.org/0000-0002-8444-8827) [44] [, B. Liberti](https://orcid.org/0000-0002-6011-2851) [71a] [, K. Lie](https://orcid.org/0000-0002-5779-5989) [60c] [, J. Lieber Marin](https://orcid.org/0000-0003-0642-9169) [78b] [, K. Lin](https://orcid.org/0000-0002-2269-3632) [103] [, R.A. Linck](https://orcid.org/0000-0002-4593-0602) [63],
R.E. Lindley [6] [, J.H. Lindon](https://orcid.org/0000-0001-9490-7276) [2] [, A. Linss](https://orcid.org/0000-0002-3961-5016) [44] [, E. Lipeles](https://orcid.org/0000-0001-5982-7326) [132] [, A. Lipniacka](https://orcid.org/0000-0002-8759-8564) [15] [, T.M. Liss](https://orcid.org/0000-0002-1735-3924) [168,ah] [, A. Lister](https://orcid.org/0000-0002-1552-3651) [170],
[J.D. Little](https://orcid.org/0000-0002-9372-0730) [7] [, B. Liu](https://orcid.org/0000-0003-2823-9307) [13a] [, B.X. Liu](https://orcid.org/0000-0002-0721-8331) [148] [, J.B. Liu](https://orcid.org/0000-0003-3259-8775) [58a] [, J.K.K. Liu](https://orcid.org/0000-0001-5359-4541) [35] [, K. Liu](https://orcid.org/0000-0001-5807-0501) [58d,58c] [, M. Liu](https://orcid.org/0000-0003-0056-7296) [58a] [, M.Y. Liu](https://orcid.org/0000-0002-0236-5404) [58a],
[P. Liu](https://orcid.org/0000-0002-9815-8898) [13a] [, X. Liu](https://orcid.org/0000-0003-1366-5530) [58a] [, Y. Liu](https://orcid.org/0000-0002-3576-7004) [44] [, Y. Liu](https://orcid.org/0000-0003-3615-2332) [13c,13d] [, Y.L. Liu](https://orcid.org/0000-0001-9190-4547) [102] [, Y.W. Liu](https://orcid.org/0000-0003-4448-4679) [58a] [, M. Livan](https://orcid.org/0000-0002-5877-0062) [68a,68b],
[J. Llorente Merino](https://orcid.org/0000-0003-0027-7969) [148] [, S.L. Lloyd](https://orcid.org/0000-0002-5073-2264) [90] [, E.M. Lobodzinska](https://orcid.org/0000-0001-9012-3431) [44] [, P. Loch](https://orcid.org/0000-0002-2005-671X) [6] [, S. Loffredo](https://orcid.org/0000-0003-2516-5015) [71a,71b] [, T. Lohse](https://orcid.org/0000-0002-9751-7633) [17],
[K. Lohwasser](https://orcid.org/0000-0003-1833-9160) [145] [, M. Lokajicek](https://orcid.org/0000-0001-8929-1243) [136] [, J.D. Long](https://orcid.org/0000-0002-2115-9382) [168] [, I. Longarini](https://orcid.org/0000-0002-0352-2854) [70a,70b] [, L. Longo](https://orcid.org/0000-0002-2357-7043) [34] [, R. Longo](https://orcid.org/0000-0003-3984-6452) [168],
I. Lopez Paz [12] [, A. Lopez Solis](https://orcid.org/0000-0002-0511-4766) [44] [, J. Lorenz](https://orcid.org/0000-0001-6530-1873) [110] [, N. Lorenzo Martinez](https://orcid.org/0000-0002-7857-7606) [4] [, A.M. Lory](https://orcid.org/0000-0001-9657-0910) [110] [, A. L¨osle](https://orcid.org/0000-0002-6328-8561) [50],
[X. Lou](https://orcid.org/0000-0002-8309-5548) [43a,43b] [, X. Lou](https://orcid.org/0000-0003-0867-2189) [13a] [, A. Lounis](https://orcid.org/0000-0003-4066-2087) [62] [, J. Love](https://orcid.org/0000-0001-7743-3849) [5] [, P.A. Love](https://orcid.org/0000-0002-7803-6674) [87] [, J.J. Lozano Bahilo](https://orcid.org/0000-0003-0613-140X) [169] [, G. Lu](https://orcid.org/0000-0001-8133-3533) [13a] [, M. Lu](https://orcid.org/0000-0001-7610-3952) [58a],
[S. Lu](https://orcid.org/0000-0002-8814-1670) [132] [, Y.J. Lu](https://orcid.org/0000-0002-2497-0509) [61] [, H.J. Lubatti](https://orcid.org/0000-0002-9285-7452) [144] [, C. Luci](https://orcid.org/0000-0001-7464-304X) [70a,70b] [, F.L. Lucio Alves](https://orcid.org/0000-0002-1626-6255) [13c] [, A. Lucotte](https://orcid.org/0000-0002-5992-0640) [56] [, F. Luehring](https://orcid.org/0000-0001-8721-6901) [63],
[I. Luise](https://orcid.org/0000-0001-5028-3342) [151], L. Luminari [70a], O. Lundberg [150] [, B. Lund-Jensen](https://orcid.org/0000-0003-3867-0336) [150] [, N.A. Luongo](https://orcid.org/0000-0001-6527-0253) [127] [, M.S. Lutz](https://orcid.org/0000-0003-4515-0224) [157] [, D. Lynn](https://orcid.org/0000-0002-9634-542X) [27],
H. Lyons [88] [, R. Lysak](https://orcid.org/0000-0003-2990-1673) [136] [, E. Lytken](https://orcid.org/0000-0002-8141-3995) [94] [, F. Lyu](https://orcid.org/0000-0002-7611-3728) [13a] [, V. Lyubushkin](https://orcid.org/0000-0003-0136-233X) [77] [, T. Lyubushkina](https://orcid.org/0000-0001-8329-7994) [77] [, H. Ma](https://orcid.org/0000-0002-8916-6220) [27] [, L.L. Ma](https://orcid.org/0000-0001-9717-1508) [58b],
[Y. Ma](https://orcid.org/0000-0002-3577-9347) [92] [, D.M. Mac Donell](https://orcid.org/0000-0001-5533-6300) [171] [, G. Maccarrone](https://orcid.org/0000-0002-7234-9522) [49] [, C.M. Macdonald](https://orcid.org/0000-0001-7857-9188) [145] [, J.C. MacDonald](https://orcid.org/0000-0002-3150-3124) [145] [, R. Madar](https://orcid.org/0000-0002-6875-6408) [36],
[W.F. Mader](https://orcid.org/0000-0003-4276-1046) [46] [, M. Madugoda Ralalage Don](https://orcid.org/0000-0002-6033-944X) [125] [, N. Madysa](https://orcid.org/0000-0001-8375-7532) [46] [, J. Maeda](https://orcid.org/0000-0002-9084-3305) [80] [, T. Maeno](https://orcid.org/0000-0003-0901-1817) [27] [, M. Maerker](https://orcid.org/0000-0002-3773-8573) [46],
[V. Magerl](https://orcid.org/0000-0003-0693-793X) [50] [, J. Magro](https://orcid.org/0000-0001-5704-9700) [64a,64c] [, D.J. Mahon](https://orcid.org/0000-0002-2640-5941) [37] [, C. Maidantchik](https://orcid.org/0000-0002-3511-0133) [78b] [, A. Maio](https://orcid.org/0000-0001-9099-0009) [135a,135b,135d] [, K. Maj](https://orcid.org/0000-0003-4819-9226) [81a],
[O. Majersky](https://orcid.org/0000-0001-8857-5770) [26a] [, S. Majewski](https://orcid.org/0000-0002-6871-3395) [127] [, N. Makovec](https://orcid.org/0000-0001-5124-904X) [62], V. Maksimovic [14] [, B. Malaescu](https://orcid.org/0000-0002-8813-3830) [131] [, Pa. Malecki](https://orcid.org/0000-0001-8183-0468) [82],
[V.P. Maleev](https://orcid.org/0000-0003-1028-8602) [133] [, F. Malek](https://orcid.org/0000-0002-0948-5775) [56] [, D. Malito](https://orcid.org/0000-0002-3996-4662) [39b,39a] [, U. Mallik](https://orcid.org/0000-0001-7934-1649) [75] [, C. Malone](https://orcid.org/0000-0003-4325-7378) [30], S. Maltezos [9], S. Malyukov [77],
[J. Mamuzic](https://orcid.org/0000-0002-3203-4243) [169] [, G. Mancini](https://orcid.org/0000-0001-6158-2751) [49] [, J.P. Mandalia](https://orcid.org/0000-0001-5038-5154) [90] [, I. Mandi´c](https://orcid.org/0000-0002-0131-7523) [89] [, L. Manhaes de Andrade Filho](https://orcid.org/0000-0003-1792-6793) [78a],
[I.M. Maniatis](https://orcid.org/0000-0002-4362-0088) [158] [, M. Manisha](https://orcid.org/0000-0001-7551-0169) [140] [, J. Manjarres Ramos](https://orcid.org/0000-0003-3896-5222) [46] [, K.H. Mankinen](https://orcid.org/0000-0001-7357-9648) [94] [, A. Mann](https://orcid.org/0000-0002-8497-9038) [110] [, A. Manousos](https://orcid.org/0000-0003-4627-4026) [74],
[B. Mansoulie](https://orcid.org/0000-0001-5945-5518) [140] [, I. Manthos](https://orcid.org/0000-0001-5561-9909) [158] [, S. Manzoni](https://orcid.org/0000-0002-2488-0511) [115] [, A. Marantis](https://orcid.org/0000-0002-7020-4098) [158,u] [, G. Marchiori](https://orcid.org/0000-0003-2655-7643) [131] [, M. Marcisovsky](https://orcid.org/0000-0003-0860-7897) [136],
[L. Marcoccia](https://orcid.org/0000-0001-6422-7018) [71a,71b] [, C. Marcon](https://orcid.org/0000-0002-9889-8271) [94] [, M. Marjanovic](https://orcid.org/0000-0002-4468-0154) [124] [, Z. Marshall](https://orcid.org/0000-0003-0786-2570) [16] [, S. Marti-Garcia](https://orcid.org/0000-0002-3897-6223) [169] [, T.A. Martin](https://orcid.org/0000-0002-1477-1645) [173],
[V.J. Martin](https://orcid.org/0000-0003-3053-8146) [48] [, B. Martin dit Latour](https://orcid.org/0000-0003-3420-2105) [15] [, L. Martinelli](https://orcid.org/0000-0002-4466-3864) [70a,70b] [, M. Martinez](https://orcid.org/0000-0002-3135-945X) [12,v] [, P. Martinez Agullo](https://orcid.org/0000-0001-8925-9518) [169],
[V.I. Martinez Outschoorn](https://orcid.org/0000-0001-7102-6388) [99] [, S. Martin-Haugh](https://orcid.org/0000-0001-9457-1928) [139] [, V.S. Martoiu](https://orcid.org/0000-0002-4963-9441) [25b] [, A.C. Martyniuk](https://orcid.org/0000-0001-9080-2944) [92] [, A. Marzin](https://orcid.org/0000-0003-4364-4351) [34],
[S.R. Maschek](https://orcid.org/0000-0003-0917-1618) [111] [, L. Masetti](https://orcid.org/0000-0002-0038-5372) [96] [, T. Mashimo](https://orcid.org/0000-0001-5333-6016) [159] [, J. Masik](https://orcid.org/0000-0002-6813-8423) [97] [, A.L. Maslennikov](https://orcid.org/0000-0002-4234-3111) [117b,117a] [, L. Massa](https://orcid.org/0000-0002-3735-7762) [21b],
[P. Massarotti](https://orcid.org/0000-0002-9335-9690) [67a,67b] [, P. Mastrandrea](https://orcid.org/0000-0002-9853-0194) [69a,69b] [, A. Mastroberardino](https://orcid.org/0000-0002-8933-9494) [39b,39a] [, T. Masubuchi](https://orcid.org/0000-0001-9984-8009) [159], D. Matakias [27],
[T. Mathisen](https://orcid.org/0000-0002-6248-953X) [167] [, A. Matic](https://orcid.org/0000-0002-2179-0350) [110], N. Matsuzawa [159] [, J. Maurer](https://orcid.org/0000-0002-5162-3713) [25b] [, B. Maˇcek](https://orcid.org/0000-0002-1449-0317) [89] [, D.A. Maximov](https://orcid.org/0000-0001-8783-3758) [117b,117a],
[R. Mazini](https://orcid.org/0000-0003-0954-0970) [154] [, I. Maznas](https://orcid.org/0000-0001-8420-3742) [158] [, S.M. Mazza](https://orcid.org/0000-0003-3865-730X) [141] [, C. Mc Ginn](https://orcid.org/0000-0003-1281-0193) [27] [, J.P. Mc Gowan](https://orcid.org/0000-0001-7551-3386) [100] [, S.P. Mc Kee](https://orcid.org/0000-0002-4551-4502) [102],
[T.G. McCarthy](https://orcid.org/0000-0002-1182-3526) [111] [, W.P. McCormack](https://orcid.org/0000-0002-0768-1959) [16] [, E.F. McDonald](https://orcid.org/0000-0002-8092-5331) [101] [, A.E. McDougall](https://orcid.org/0000-0002-2489-2598) [115] [, J.A. Mcfayden](https://orcid.org/0000-0001-9273-2564) [152],
[G. Mchedlidze](https://orcid.org/0000-0003-3534-4164) [155b], M.A. McKay [40] [, K.D. McLean](https://orcid.org/0000-0001-5475-2521) [171] [, S.J. McMahon](https://orcid.org/0000-0002-3599-9075) [139] [, P.C. McNamara](https://orcid.org/0000-0002-0676-324X) [101],
[R.A. McPherson](https://orcid.org/0000-0001-9211-7019) [171,y] [, J.E. Mdhluli](https://orcid.org/0000-0002-9745-0504) [31f] [, Z.A. Meadows](https://orcid.org/0000-0001-8119-0333) [99] [, S. Meehan](https://orcid.org/0000-0002-3613-7514) [34] [, T. Megy](https://orcid.org/0000-0001-8569-7094) [36] [, S. Mehlhase](https://orcid.org/0000-0002-1281-2060) [110],
[A. Mehta](https://orcid.org/0000-0003-2619-9743) [88] [, B. Meirose](https://orcid.org/0000-0003-0032-7022) [41] [, D. Melini](https://orcid.org/0000-0002-7018-682X) [156] [, B.R. Mellado Garcia](https://orcid.org/0000-0003-4838-1546) [31f] [, A.H. Melo](https://orcid.org/0000-0002-3964-6736) [51] [, F. Meloni](https://orcid.org/0000-0001-7075-2214) [44] [, A. Melzer](https://orcid.org/0000-0002-7616-3290) [22],
[E.D. Mendes Gouveia](https://orcid.org/0000-0002-7785-2047) [135a] [, A.M. Mendes Jacques Da Costa](https://orcid.org/0000-0001-6305-8400) [19], H.Y. Meng [162] [, L. Meng](https://orcid.org/0000-0002-2901-6589) [34] [, S. Menke](https://orcid.org/0000-0002-8186-4032) [111],
[M. Mentink](https://orcid.org/0000-0001-9769-0578) [34] [, E. Meoni](https://orcid.org/0000-0002-6934-3752) [39b,39a] [, C. Merlassino](https://orcid.org/0000-0002-5445-5938) [130] [, P. Mermod](https://orcid.org/0000-0001-9656-9901) [52,*] [, L. Merola](https://orcid.org/0000-0002-1822-1114) [67a,67b] [, C. Meroni](https://orcid.org/0000-0003-4779-3522) [66a],
G. Merz [102] [, O. Meshkov](https://orcid.org/0000-0001-6897-4651) [107,109] [, J.K.R. Meshreki](https://orcid.org/0000-0003-2007-7171) [147] [, J. Metcalfe](https://orcid.org/0000-0001-5454-3017) [5] [, A.S. Mete](https://orcid.org/0000-0002-5508-530X) [5] [, C. Meyer](https://orcid.org/0000-0003-3552-6566) [63] [, J-P. Meyer](https://orcid.org/0000-0002-7497-0945) [140],
[M. Michetti](https://orcid.org/0000-0002-3276-8941) [17] [, R.P. Middleton](https://orcid.org/0000-0002-8396-9946) [139] [, L. Mijovi´c](https://orcid.org/0000-0003-0162-2891) [48] [, G. Mikenberg](https://orcid.org/0000-0003-0460-3178) [175] [, M. Mikestikova](https://orcid.org/0000-0003-1277-2596) [136] [, M. Mikuˇz](https://orcid.org/0000-0002-4119-6156) [89],
[H. Mildner](https://orcid.org/0000-0002-0384-6955) [145] [, A. Milic](https://orcid.org/0000-0002-9173-8363) [162] [, C.D. Milke](https://orcid.org/0000-0003-4688-4174) [40] [, D.W. Miller](https://orcid.org/0000-0002-9485-9435) [35] [, L.S. Miller](https://orcid.org/0000-0001-5539-3233) [32] [, A. Milov](https://orcid.org/0000-0003-3863-3607) [175], D.A. Milstead [43a,43b],
T. Min [13c] [, A.A. Minaenko](https://orcid.org/0000-0001-8055-4692) [118] [, I.A. Minashvili](https://orcid.org/0000-0002-4688-3510) [155b] [, L. Mince](https://orcid.org/0000-0003-3759-0588) [55] [, A.I. Mincer](https://orcid.org/0000-0002-6307-1418) [121] [, B. Mindur](https://orcid.org/0000-0002-5511-2611) [81a] [, M. Mineev](https://orcid.org/0000-0002-2236-3879) [77],
Y. Minegishi [159] [, Y. Mino](https://orcid.org/0000-0002-2984-8174) [83] [, L.M. Mir](https://orcid.org/0000-0002-4276-715X) [12] [, M. Miralles Lopez](https://orcid.org/0000-0001-7863-583X) [169] [, M. Mironova](https://orcid.org/0000-0001-6381-5723) [130] [, T. Mitani](https://orcid.org/0000-0001-9861-9140) [174],
[V.A. Mitsou](https://orcid.org/0000-0002-1533-8886) [169], M. Mittal [58c] [, O. Miu](https://orcid.org/0000-0002-0287-8293) [162] [, P.S. Miyagawa](https://orcid.org/0000-0002-4893-6778) [90], Y. Miyazaki [85] [, A. Mizukami](https://orcid.org/0000-0001-6672-0500) [79],
[J.U. Mj¨ornmark](https://orcid.org/0000-0002-7148-6859) [94] [, T. Mkrtchyan](https://orcid.org/0000-0002-5786-3136) [59a] [, M. Mlynarikova](https://orcid.org/0000-0003-2028-1930) [116] [, T. Moa](https://orcid.org/0000-0002-7644-5984) [43a,43b] [, S. Mobius](https://orcid.org/0000-0001-5911-6815) [51] [, K. Mochizuki](https://orcid.org/0000-0002-6310-2149) [106],
[P. Moder](https://orcid.org/0000-0003-2135-9971) [44] [, P. Mogg](https://orcid.org/0000-0003-2688-234X) [110] [, A.F. Mohammed](https://orcid.org/0000-0002-5003-1919) [13a] [, S. Mohapatra](https://orcid.org/0000-0003-3006-6337) [37] [, G. Mokgatitswane](https://orcid.org/0000-0001-9878-4373) [31f] [, B. Mondal](https://orcid.org/0000-0003-1025-3741) [147],
[S. Mondal](https://orcid.org/0000-0002-6965-7380) [137] [, K. M¨onig](https://orcid.org/0000-0002-3169-7117) [44] [, E. Monnier](https://orcid.org/0000-0002-2551-5751) [98], L. Monsonis Romero [169] [, A. Montalbano](https://orcid.org/0000-0002-5295-432X) [148],
[J. Montejo Berlingen](https://orcid.org/0000-0001-9213-904X) [34] [, M. Montella](https://orcid.org/0000-0001-5010-886X) [123] [, F. Monticelli](https://orcid.org/0000-0002-6974-1443) [86] [, N. Morange](https://orcid.org/0000-0003-0047-7215) [62] [, A.L. Moreira De Carvalho](https://orcid.org/0000-0002-1986-5720) [135a],
[M. Moreno Llacer](https://orcid.org/0000-0003-1113-3645) ´ [169] [, C. Moreno Martinez](https://orcid.org/0000-0002-5719-7655) [12] [, P. Morettini](https://orcid.org/0000-0001-7139-7912) [53b] [, S. Morgenstern](https://orcid.org/0000-0002-7834-4781) [173] [, D. Mori](https://orcid.org/0000-0002-0693-4133) [148] [, M. Morii](https://orcid.org/0000-0001-9324-057X) [57],
[M. Morinaga](https://orcid.org/0000-0003-2129-1372) [159] [, V. Morisbak](https://orcid.org/0000-0001-8715-8780) [129] [, A.K. Morley](https://orcid.org/0000-0003-0373-1346) [34] [, D. Morozova](https://orcid.org/0000-0002-5118-9769) [48] [, A.P. Morris](https://orcid.org/0000-0002-2929-3869) [92] [, L. Morvaj](https://orcid.org/0000-0003-2061-2904) [34],


64


[P. Moschovakos](https://orcid.org/0000-0001-6993-9698) [34] [, B. Moser](https://orcid.org/0000-0001-6750-5060) [115], M. Mosidze [155b] [, T. Moskalets](https://orcid.org/0000-0001-6508-3968) [50] [, P. Moskvitina](https://orcid.org/0000-0002-7926-7650) [114] [, J. Moss](https://orcid.org/0000-0002-6729-4803) [29,n],
[E.J.W. Moyse](https://orcid.org/0000-0003-4449-6178) [99] [, S. Muanza](https://orcid.org/0000-0002-1786-2075) [98] [, J. Mueller](https://orcid.org/0000-0001-5099-4718) [134] [, R. Mueller](https://orcid.org/0000-0002-5835-0690) [18] [, D. Muenstermann](https://orcid.org/0000-0001-6223-2497) [87] [, G.A. Mullier](https://orcid.org/0000-0001-6771-0937) [94],
J.J. Mullin [132] [, D.P. Mungo](https://orcid.org/0000-0002-2567-7857) [66a,66b] [, J.L. Munoz Martinez](https://orcid.org/0000-0002-2441-3366) [12] [, F.J. Munoz Sanchez](https://orcid.org/0000-0002-6374-458X) [97] [, M. Murin](https://orcid.org/0000-0002-2388-1969) [97] [, P. Murin](https://orcid.org/0000-0001-9686-2139) [26b],
[W.J. Murray](https://orcid.org/0000-0003-1710-6306) [173,139] [, A. Murrone](https://orcid.org/0000-0001-5399-2478) [66a,66b] [, J.M. Muse](https://orcid.org/0000-0002-2585-3793) [124] [, M. Muˇskinja](https://orcid.org/0000-0001-8442-2718) [16] [, C. Mwewa](https://orcid.org/0000-0002-3504-0366) [27] [, A.G. Myagkov](https://orcid.org/0000-0003-4189-4250) [118,ae],
[A.J. Myers](https://orcid.org/0000-0003-1691-4643) [7], A.A. Myers [134] [, G. Myers](https://orcid.org/0000-0002-2562-0930) [63] [, M. Myska](https://orcid.org/0000-0003-0982-3380) [137] [, B.P. Nachman](https://orcid.org/0000-0003-1024-0932) [16] [, O. Nackenhorst](https://orcid.org/0000-0002-2191-2725) [45] [, A.Nag Nag](https://orcid.org/0000-0001-6480-6079) [46],
[K. Nagai](https://orcid.org/0000-0002-4285-0578) [130] [, K. Nagano](https://orcid.org/0000-0003-2741-0627) [79] [, J.L. Nagle](https://orcid.org/0000-0003-0056-6613) [27] [, E. Nagy](https://orcid.org/0000-0001-5420-9537) [98] [, A.M. Nairz](https://orcid.org/0000-0003-3561-0880) [34] [, Y. Nakahama](https://orcid.org/0000-0003-3133-7100) [112] [, K. Nakamura](https://orcid.org/0000-0002-1560-0434) [79],
[H. Nanjo](https://orcid.org/0000-0003-0703-103X) [128] [, F. Napolitano](https://orcid.org/0000-0002-8686-5923) [59a] [, R. Narayan](https://orcid.org/0000-0002-8642-5119) [40] [, E.A. Narayanan](https://orcid.org/0000-0001-6042-6781) [113] [, I. Naryshkin](https://orcid.org/0000-0001-6412-4801) [133] [, M. Naseri](https://orcid.org/0000-0001-9191-8164) [32] [, C. Nass](https://orcid.org/0000-0002-8098-4948) [22],
[T. Naumann](https://orcid.org/0000-0001-7372-8316) [44] [, G. Navarro](https://orcid.org/0000-0002-5108-0042) [20a] [, J. Navarro-Gonzalez](https://orcid.org/0000-0002-4172-7965) [169] [, R. Nayak](https://orcid.org/0000-0001-6988-0606) [157] [, P.Y. Nechaeva](https://orcid.org/0000-0002-5910-4117) [107] [, F. Nechansky](https://orcid.org/0000-0002-2684-9024) [44],
[T.J. Neep](https://orcid.org/0000-0003-0056-8651) [19] [, A. Negri](https://orcid.org/0000-0002-7386-901X) [68a,68b] [, M. Negrini](https://orcid.org/0000-0003-0101-6963) [21b] [, C. Nellist](https://orcid.org/0000-0002-5171-8579) [114] [, C. Nelson](https://orcid.org/0000-0002-5713-3803) [100] [, K. Nelson](https://orcid.org/0000-0003-4194-1790) [102] [, S. Nemecek](https://orcid.org/0000-0001-8978-7150) [136],
[M. Nessi](https://orcid.org/0000-0001-7316-0118) [34,f] [, M.S. Neubauer](https://orcid.org/0000-0001-8434-9274) [168] [, F. Neuhaus](https://orcid.org/0000-0002-3819-2453) [96] [, J. Neundorf](https://orcid.org/0000-0002-8565-0015) [44] [, R. Newhouse](https://orcid.org/0000-0001-8026-3836) [170] [, P.R. Newman](https://orcid.org/0000-0002-6252-266X) [19],
[C.W. Ng](https://orcid.org/0000-0001-8190-4017) [134], Y.S. Ng [17] [, Y.W.Y. Ng](https://orcid.org/0000-0001-9135-1321) [166] [, B. Ngair](https://orcid.org/0000-0002-5807-8535) [33e] [, H.D.N. Nguyen](https://orcid.org/0000-0002-4326-9283) [106] [, R.B. Nickerson](https://orcid.org/0000-0002-2157-9061) [130],
[R. Nicolaidou](https://orcid.org/0000-0003-3723-1745) [140] [, D.S. Nielsen](https://orcid.org/0000-0002-9341-6907) [38] [, J. Nielsen](https://orcid.org/0000-0002-9175-4419) [141] [, M. Niemeyer](https://orcid.org/0000-0003-4222-8284) [51] [, N. Nikiforou](https://orcid.org/0000-0003-1267-7740) [10] [, V. Nikolaenko](https://orcid.org/0000-0001-6545-1820) [118,ae],
[I. Nikolic-Audit](https://orcid.org/0000-0003-1681-1118) [131] [, K. Nikolopoulos](https://orcid.org/0000-0002-3048-489X) [19] [, P. Nilsson](https://orcid.org/0000-0002-6848-7463) [27] [, H.R. Nindhito](https://orcid.org/0000-0003-3108-9477) [52] [, A. Nisati](https://orcid.org/0000-0002-5080-2293) [70a] [, N. Nishu](https://orcid.org/0000-0002-9048-1332) [2] [, R. Nisius](https://orcid.org/0000-0003-2257-0074) [111],
[T. Nitta](https://orcid.org/0000-0002-9234-4833) [174] [, T. Nobe](https://orcid.org/0000-0002-5809-325X) [159] [, D.L. Noel](https://orcid.org/0000-0001-8889-427X) [30] [, Y. Noguchi](https://orcid.org/0000-0002-3113-3127) [83] [, I. Nomidis](https://orcid.org/0000-0002-7406-1100) [131], M.A. Nomura [27] [, M.B. Norfolk](https://orcid.org/0000-0001-7984-5783) [145],
[R.R.B. Norisam](https://orcid.org/0000-0002-4129-5736) [92] [, J. Novak](https://orcid.org/0000-0002-3195-8903) [89] [, T. Novak](https://orcid.org/0000-0002-3053-0913) [44] [, O. Novgorodova](https://orcid.org/0000-0001-6536-0179) [46] [, L. Novotny](https://orcid.org/0000-0001-5165-8425) [137] [, R. Novotny](https://orcid.org/0000-0002-1630-694X) [113], L. Nozka [126],
[K. Ntekas](https://orcid.org/0000-0001-9252-6509) [166], E. Nurse [92] [, F.G. Oakham](https://orcid.org/0000-0003-2866-1049) [32,ai] [, J. Ocariz](https://orcid.org/0000-0003-2262-0780) [131] [, A. Ochi](https://orcid.org/0000-0002-2024-5609) [80] [, I. Ochoa](https://orcid.org/0000-0001-6156-1790) [135a] [, J.P. Ochoa-Ricoux](https://orcid.org/0000-0001-7376-5555) [142a],
[S. Oda](https://orcid.org/0000-0001-5836-768X) [85] [, S. Odaka](https://orcid.org/0000-0002-1227-1401) [79] [, S. Oerdek](https://orcid.org/0000-0001-8763-0096) [167] [, A. Ogrodnik](https://orcid.org/0000-0002-6025-4833) [81a] [, A. Oh](https://orcid.org/0000-0001-9025-0422) [97] [, C.C. Ohm](https://orcid.org/0000-0002-8015-7512) [150] [, H. Oide](https://orcid.org/0000-0002-2173-3233) [160] [, R. Oishi](https://orcid.org/0000-0001-6930-7789) [159],
[M.L. Ojeda](https://orcid.org/0000-0002-3834-7830) [44] [, Y. Okazaki](https://orcid.org/0000-0003-2677-5827) [83], M.W. O’Keefe [88] [, Y. Okumura](https://orcid.org/0000-0002-7613-5572) [159], A. Olariu [25b] [, L.F. Oleiro Seabra](https://orcid.org/0000-0002-9320-8825) [135a],
[S.A. Olivares Pino](https://orcid.org/0000-0003-4616-6973) [142d] [, D. Oliveira Damazio](https://orcid.org/0000-0002-8601-2074) [27] [, D. Oliveira Goncalves](https://orcid.org/0000-0002-1943-9561) [78a] [, J.L. Oliver](https://orcid.org/0000-0002-0713-6627) [166] [, M.J.R. Olsson](https://orcid.org/0000-0003-4154-8139) [166],
[A. Olszewski](https://orcid.org/0000-0003-3368-5475) [82] [, J. Olszowska](https://orcid.org/0000-0003-0520-9500) [82] [, O.O.](https://orcid.org/0000-0001-8772-1705) [¨] [Oncel](https://orcid.org/0000-0001-8772-1705) [¨] [22] [, D.C. O’Neil](https://orcid.org/0000-0003-0325-472X) [148] [, A.P. O’neill](https://orcid.org/0000-0002-8104-7227) [130] [, A. Onofre](https://orcid.org/0000-0003-3471-2703) [135a,135e],
[P.U.E. Onyisi](https://orcid.org/0000-0003-4201-7997) [10], R.G. Oreamuno Madriz [116] [, M.J. Oreglia](https://orcid.org/0000-0001-6203-2209) [35] [, G.E. Orellana](https://orcid.org/0000-0002-4753-4048) [86] [, D. Orestano](https://orcid.org/0000-0001-5103-5527) [72a,72b],
[N. Orlando](https://orcid.org/0000-0003-0616-245X) [12] [, R.S. Orr](https://orcid.org/0000-0002-8690-9746) [162] [, V. O’Shea](https://orcid.org/0000-0001-7183-1205) [55] [, R. Ospanov](https://orcid.org/0000-0001-5091-9216) [58a] [, G. Otero y Garzon](https://orcid.org/0000-0003-4803-5280) [28] [, H. Otono](https://orcid.org/0000-0003-0760-5988) [85] [, P.S. Ott](https://orcid.org/0000-0003-1052-7925) [59a],
[G.J. Ottino](https://orcid.org/0000-0001-8083-6411) [16] [, M. Ouchrif](https://orcid.org/0000-0002-2954-1420) [33d] [, J. Ouellette](https://orcid.org/0000-0002-0582-3765) [27] [, F. Ould-Saada](https://orcid.org/0000-0002-9404-835X) [129] [, A. Ouraou](https://orcid.org/0000-0001-6818-5994) [140,*] [, Q. Ouyang](https://orcid.org/0000-0002-8186-0082) [13a] [, M. Owen](https://orcid.org/0000-0001-6820-0488) [55],
[R.E. Owen](https://orcid.org/0000-0002-2684-1399) [139] [, K.Y. Oyulmaz](https://orcid.org/0000-0002-5533-9621) [11c] [, V.E. Ozcan](https://orcid.org/0000-0003-4643-6347) [11c] [, N. Ozturk](https://orcid.org/0000-0003-1125-6784) [7] [, S. Ozturk](https://orcid.org/0000-0001-6533-6144) [11c] [, J. Pacalt](https://orcid.org/0000-0002-0148-7207) [126] [, H.A. Pacey](https://orcid.org/0000-0002-2325-6792) [30],
[K. Pachal](https://orcid.org/0000-0002-8332-243X) [47] [, A. Pacheco Pages](https://orcid.org/0000-0001-8210-1734) [12] [, C. Padilla Aranda](https://orcid.org/0000-0001-7951-0166) [12] [, S. Pagan Griso](https://orcid.org/0000-0003-0999-5019) [16] [, G. Palacino](https://orcid.org/0000-0003-0278-9941) [63] [, S. Palazzo](https://orcid.org/0000-0002-4225-387X) [48],
[S. Palestini](https://orcid.org/0000-0002-4110-096X) [34] [, M. Palka](https://orcid.org/0000-0002-7185-3540) [81b] [, P. Palni](https://orcid.org/0000-0001-6201-2785) [81a] [, D.K. Panchal](https://orcid.org/0000-0001-5732-9948) [10] [, C.E. Pandini](https://orcid.org/0000-0003-3838-1307) [52] [, J.G. Panduro Vazquez](https://orcid.org/0000-0003-2605-8940) [91] [, P. Pani](https://orcid.org/0000-0003-2149-3791) [44],
[G. Panizzo](https://orcid.org/0000-0002-0352-4833) [64a,64c] [, L. Paolozzi](https://orcid.org/0000-0002-9281-1972) [52] [, C. Papadatos](https://orcid.org/0000-0003-3160-3077) [106] [, S. Parajuli](https://orcid.org/0000-0003-1499-3990) [40] [, A. Paramonov](https://orcid.org/0000-0002-6492-3061) [5] [, C. Paraskevopoulos](https://orcid.org/0000-0002-2858-9182) [9],
[D. Paredes Hernandez](https://orcid.org/0000-0002-3179-8524) [60b] [, S.R. Paredes Saenz](https://orcid.org/0000-0001-8487-9603) [130] [, B. Parida](https://orcid.org/0000-0001-9367-8061) [175] [, T.H. Park](https://orcid.org/0000-0002-1910-0541) [162] [, A.J. Parker](https://orcid.org/0000-0001-9410-3075) [29] [, M.A. Parker](https://orcid.org/0000-0001-9798-8411) [30],
[F. Parodi](https://orcid.org/0000-0002-7160-4720) [53b,53a] [, E.W. Parrish](https://orcid.org/0000-0001-5954-0974) [116] [, J.A. Parsons](https://orcid.org/0000-0002-9470-6017) [37] [, U. Parzefall](https://orcid.org/0000-0002-4858-6560) [50] [, L. Pascual Dominguez](https://orcid.org/0000-0003-4701-9481) [157] [, V.R. Pascuzzi](https://orcid.org/0000-0003-3167-8773) [16],
[F. Pasquali](https://orcid.org/0000-0003-0707-7046) [115] [, E. Pasqualucci](https://orcid.org/0000-0001-8160-2545) [70a] [, S. Passaggio](https://orcid.org/0000-0001-9200-5738) [53b] [, F. Pastore](https://orcid.org/0000-0001-5962-7826) [91] [, P. Pasuwan](https://orcid.org/0000-0003-2987-2964) [43a,43b] [, J.R. Pater](https://orcid.org/0000-0002-0598-5035) [97],
[A. Pathak](https://orcid.org/0000-0001-9861-2942) [176], J. Patton [88] [, T. Pauly](https://orcid.org/0000-0001-9082-035X) [34] [, J. Pearkes](https://orcid.org/0000-0002-5205-4065) [149] [, M. Pedersen](https://orcid.org/0000-0003-4281-0119) [129] [, L. Pedraza Diaz](https://orcid.org/0000-0003-3924-8276) [114] [, R. Pedro](https://orcid.org/0000-0002-7139-9587) [135a],
[T. Peiffer](https://orcid.org/0000-0002-8162-6667) [51] [, S.V. Peleganchuk](https://orcid.org/0000-0003-0907-7592) [117b,117a] [, O. Penc](https://orcid.org/0000-0002-5433-3981) [136] [, C. Peng](https://orcid.org/0000-0002-3451-2237) [60b] [, H. Peng](https://orcid.org/0000-0002-3461-0945) [58a] [, M. Penzin](https://orcid.org/0000-0002-0928-3129) [161] [, B.S. Peralva](https://orcid.org/0000-0003-1664-5658) [78a],
[A.P. Pereira Peixoto](https://orcid.org/0000-0003-3424-7338) [135a] [, L. Pereira Sanchez](https://orcid.org/0000-0001-7913-3313) [43a,43b] [, D.V. Perepelitsa](https://orcid.org/0000-0001-8732-6908) [27] [, E. Perez Codina](https://orcid.org/0000-0003-0426-6538) [163a] [, M. Perganti](https://orcid.org/0000-0003-3451-9938) [9],
[L. Perini](https://orcid.org/0000-0003-3715-0523) [66a,66b] [, H. Pernegger](https://orcid.org/0000-0001-6418-8784) [34] [, S. Perrella](https://orcid.org/0000-0003-4955-5130) [34] [, A. Perrevoort](https://orcid.org/0000-0001-6343-447X) [115] [, K. Peters](https://orcid.org/0000-0002-7654-1677) [44] [, R.F.Y. Peters](https://orcid.org/0000-0003-1702-7544) [97],
[B.A. Petersen](https://orcid.org/0000-0002-7380-6123) [34] [, T.C. Petersen](https://orcid.org/0000-0003-0221-3037) [38] [, E. Petit](https://orcid.org/0000-0002-3059-735X) [98] [, V. Petousis](https://orcid.org/0000-0002-5575-6476) [137] [, C. Petridou](https://orcid.org/0000-0001-5957-6133) [158], P. Petroff [62] [, F. Petrucci](https://orcid.org/0000-0002-5278-2206) [72a,72b],
[A. Petrukhin](https://orcid.org/0000-0003-0533-2277) [147] [, M. Pettee](https://orcid.org/0000-0001-9208-3218) [178] [, N.E. Pettersson](https://orcid.org/0000-0001-7451-3544) [34] [, K. Petukhova](https://orcid.org/0000-0002-0654-8398) [138] [, A. Peyaud](https://orcid.org/0000-0001-8933-8689) [140] [, R. Pezoa](https://orcid.org/0000-0003-3344-791X) [142e],
[L. Pezzotti](https://orcid.org/0000-0002-3802-8944) [34] [, G. Pezzullo](https://orcid.org/0000-0002-6653-1555) [178] [, T. Pham](https://orcid.org/0000-0002-8859-1313) [101] [, P.W. Phillips](https://orcid.org/0000-0003-3651-4081) [139] [, M.W. Phipps](https://orcid.org/0000-0002-5367-8961) [168] [, G. Piacquadio](https://orcid.org/0000-0002-4531-2900) [151] [, E. Pianori](https://orcid.org/0000-0001-9233-5892) [16],
[F. Piazza](https://orcid.org/0000-0002-3664-8912) [66a,66b] [, A. Picazio](https://orcid.org/0000-0001-5070-4717) [99] [, R. Piegaia](https://orcid.org/0000-0001-7850-8005) [28] [, D. Pietreanu](https://orcid.org/0000-0003-1381-5949) [25b] [, J.E. Pilcher](https://orcid.org/0000-0003-2417-2176) [35] [, A.D. Pilkington](https://orcid.org/0000-0001-8007-0778) [97],
[M. Pinamonti](https://orcid.org/0000-0002-5282-5050) [64a,64c] [, J.L. Pinfold](https://orcid.org/0000-0002-2397-4196) [2], C. Pitman Donaldson [92] [, D.A. Pizzi](https://orcid.org/0000-0001-5193-1567) [32] [, L. Pizzimento](https://orcid.org/0000-0002-1814-2758) [71a,71b],
[A. Pizzini](https://orcid.org/0000-0001-8891-1842) [115] [, M.-A. Pleier](https://orcid.org/0000-0002-9461-3494) [27], V. Plesanovs [50] [, V. Pleskot](https://orcid.org/0000-0001-5435-497X) [138], E. Plotnikova [77] [, P. Podberezko](https://orcid.org/0000-0002-1142-3215) [117b,117a],
[R. Poettgen](https://orcid.org/0000-0002-3304-0987) [94] [, R. Poggi](https://orcid.org/0000-0002-7324-9320) [52] [, L. Poggioli](https://orcid.org/0000-0003-3210-6646) [131] [, I. Pogrebnyak](https://orcid.org/0000-0002-3817-0879) [103] [, D. Pohl](https://orcid.org/0000-0002-3332-1113) [22] [, I. Pokharel](https://orcid.org/0000-0002-7915-0161) [51] [, G. Polesello](https://orcid.org/0000-0001-8636-0186) [68a],
[A. Poley](https://orcid.org/0000-0002-4063-0408) [148,163a] [, A. Policicchio](https://orcid.org/0000-0002-1290-220X) [70a,70b] [, R. Polifka](https://orcid.org/0000-0003-1036-3844) [138] [, A. Polini](https://orcid.org/0000-0002-4986-6628) [21b] [, C.S. Pollard](https://orcid.org/0000-0002-3690-3960) [130] [, Z.B. Pollock](https://orcid.org/0000-0001-6285-0658) [123],
[V. Polychronakos](https://orcid.org/0000-0002-4051-0828) [27] [, D. Ponomarenko](https://orcid.org/0000-0003-4213-1511) [108] [, L. Pontecorvo](https://orcid.org/0000-0003-2284-3765) [34] [, S. Popa](https://orcid.org/0000-0001-9275-4536) [25a] [, G.A. Popeneciu](https://orcid.org/0000-0001-9783-7736) [25d] [, L. Portales](https://orcid.org/0000-0002-9860-9185) [4],
[D.M. Portillo Quintero](https://orcid.org/0000-0002-7042-4058) [163a] [, S. Pospisil](https://orcid.org/0000-0001-5424-9096) [137] [, P. Postolache](https://orcid.org/0000-0001-8797-012X) [25c] [, K. Potamianos](https://orcid.org/0000-0001-7839-9785) [130] [, I.N. Potrap](https://orcid.org/0000-0002-0375-6909) [77] [, C.J. Potter](https://orcid.org/0000-0002-9815-5208) [30],
[H. Potti](https://orcid.org/0000-0002-0800-9902) [1] [, T. Poulsen](https://orcid.org/0000-0001-7207-6029) [44] [, J. Poveda](https://orcid.org/0000-0001-8144-1964) [169] [, T.D. Powell](https://orcid.org/0000-0001-9381-7850) [145] [, G. Pownall](https://orcid.org/0000-0002-9244-0753) [44] [, M.E. Pozo Astigarraga](https://orcid.org/0000-0002-3069-3077) [34],
[A. Prades Ibanez](https://orcid.org/0000-0003-1418-2012) [169] [, P. Pralavorio](https://orcid.org/0000-0002-2452-6715) [98] [, M.M. Prapa](https://orcid.org/0000-0001-6778-9403) [42] [, S. Prell](https://orcid.org/0000-0002-0195-8005) [76] [, D. Price](https://orcid.org/0000-0003-2750-9977) [97] [, M. Primavera](https://orcid.org/0000-0002-6866-3818) [65a],
[M.A. Principe Martin](https://orcid.org/0000-0002-5085-2717) [95] [, M.L. Proffitt](https://orcid.org/0000-0003-0323-8252) [144] [, N. Proklova](https://orcid.org/0000-0002-5237-0201) [108] [, K. Prokofiev](https://orcid.org/0000-0002-2177-6401) [60c] [, F. Prokoshin](https://orcid.org/0000-0001-6389-5399) [77],


65


[S. Protopopescu](https://orcid.org/0000-0001-7432-8242) [27] [, J. Proudfoot](https://orcid.org/0000-0003-1032-9945) [5] [, M. Przybycien](https://orcid.org/0000-0002-9235-2649) [81a] [, D. Pudzha](https://orcid.org/0000-0002-7026-1412) [133], P. Puzo [62] [, D. Pyatiizbyantseva](https://orcid.org/0000-0002-6659-8506) [108],
[J. Qian](https://orcid.org/0000-0003-4813-8167) [102] [, Y. Qin](https://orcid.org/0000-0002-6960-502X) [97] [, T. Qiu](https://orcid.org/0000-0001-5047-3031) [90] [, A. Quadt](https://orcid.org/0000-0002-0098-384X) [51] [, M. Queitsch-Maitland](https://orcid.org/0000-0003-4643-515X) [34] [, G. Rabanal Bolanos](https://orcid.org/0000-0003-1526-5848) [57],
[F. Ragusa](https://orcid.org/0000-0002-4064-0489) [66a,66b] [, J.A. Raine](https://orcid.org/0000-0002-5987-4648) [52] [, S. Rajagopalan](https://orcid.org/0000-0001-6543-1520) [27] [, K. Ran](https://orcid.org/0000-0003-3119-9924) [13a,13d] [, D.F. Rassloff](https://orcid.org/0000-0002-5756-4558) [59a] [, D.M. Rauch](https://orcid.org/0000-0002-8527-7695) [44] [, S. Rave](https://orcid.org/0000-0002-0050-8053) [96],
[B. Ravina](https://orcid.org/0000-0002-1622-6640) [55] [, I. Ravinovich](https://orcid.org/0000-0001-9348-4363) [175] [, M. Raymond](https://orcid.org/0000-0001-8225-1142) [34] [, A.L. Read](https://orcid.org/0000-0002-5751-6636) [129] [, N.P. Readioff](https://orcid.org/0000-0002-3427-0688) [145] [, D.M. Rebuzzi](https://orcid.org/0000-0003-4461-3880) [68a,68b],
[G. Redlinger](https://orcid.org/0000-0002-6437-9991) [27] [, K. Reeves](https://orcid.org/0000-0003-3504-4882) [41] [, D. Reikher](https://orcid.org/0000-0001-5758-579X) [157], A. Reiss [96] [, A. Rej](https://orcid.org/0000-0002-5471-0118) [147] [, C. Rembser](https://orcid.org/0000-0001-6139-2210) [34] [, A. Renardi](https://orcid.org/0000-0003-4021-6482) [44],
[M. Renda](https://orcid.org/0000-0002-0429-6959) [25b], M.B. Rendel [111] [, A.G. Rennie](https://orcid.org/0000-0002-8485-3734) [55] [, S. Resconi](https://orcid.org/0000-0003-2313-4020) [66a] [, M. Ressegotti](https://orcid.org/0000-0002-6777-1761) [53b,53a] [, E.D. Resseguie](https://orcid.org/0000-0002-7739-6176) [16],
[S. Rettie](https://orcid.org/0000-0002-7092-3893) [92], B. Reynolds [123] [, E. Reynolds](https://orcid.org/0000-0002-1506-5750) [19] [, M. Rezaei Estabragh](https://orcid.org/0000-0002-3308-8067) [177] [, O.L. Rezanova](https://orcid.org/0000-0001-7141-0304) [117b,117a],
[P. Reznicek](https://orcid.org/0000-0003-4017-9829) [138] [, E. Ricci](https://orcid.org/0000-0002-4222-9976) [73a,73b] [, R. Richter](https://orcid.org/0000-0001-8981-1966) [111] [, S. Richter](https://orcid.org/0000-0001-6613-4448) [44] [, E. Richter-Was](https://orcid.org/0000-0002-3823-9039) [81b] [, M. Ridel](https://orcid.org/0000-0002-2601-7420) [131] [, P. Rieck](https://orcid.org/0000-0003-0290-0566) [111],
[P. Riedler](https://orcid.org/0000-0002-4871-8543) [34] [, O. Rifki](https://orcid.org/0000-0002-9169-0793) [44] [, M. Rijssenbeek](https://orcid.org/0000-0002-3476-1575) [151] [, A. Rimoldi](https://orcid.org/0000-0003-3590-7908) [68a,68b] [, M. Rimoldi](https://orcid.org/0000-0003-1165-7940) [44] [, L. Rinaldi](https://orcid.org/0000-0001-9608-9940) [21b,21a],
[T.T. Rinn](https://orcid.org/0000-0002-1295-1538) [168] [, M.P. Rinnagel](https://orcid.org/0000-0003-4931-0459) [110] [, G. Ripellino](https://orcid.org/0000-0002-4053-5144) [150] [, I. Riu](https://orcid.org/0000-0002-3742-4582) [12] [, P. Rivadeneira](https://orcid.org/0000-0002-7213-3844) [44] [, J.C. Rivera Vergara](https://orcid.org/0000-0002-8149-4561) [171],
[F. Rizatdinova](https://orcid.org/0000-0002-2041-6236) [125] [, E. Rizvi](https://orcid.org/0000-0001-9834-2671) [90] [, C. Rizzi](https://orcid.org/0000-0001-6120-2325) [52] [, B.A. Roberts](https://orcid.org/0000-0001-5904-0582) [173] [, B.R. Roberts](https://orcid.org/0000-0001-5235-8256) [16] [, S.H. Robertson](https://orcid.org/0000-0003-4096-8393) [100,y],
[M. Robin](https://orcid.org/0000-0002-1390-7141) [44] [, D. Robinson](https://orcid.org/0000-0001-6169-4868) [30], C.M. Robles Gajardo [142e] [, M. Robles Manzano](https://orcid.org/0000-0001-7701-8864) [96] [, A. Robson](https://orcid.org/0000-0002-1659-8284) [55],
[A. Rocchi](https://orcid.org/0000-0002-3125-8333) [71a,71b] [, C. Roda](https://orcid.org/0000-0002-3020-4114) [69a,69b] [, S. Rodriguez Bosca](https://orcid.org/0000-0002-4571-2509) [59a] [, A. Rodriguez Rodriguez](https://orcid.org/0000-0002-1590-2352) [50],
[A.M. Rodr](https://orcid.org/0000-0002-9609-3306) ´ ıguez Vera [163b], S. Roe [34] [, A.R. Roepe](https://orcid.org/0000-0001-5933-9357) [124] [, J. Roggel](https://orcid.org/0000-0002-5749-3876) [177] [, O. Røhne](https://orcid.org/0000-0001-7744-9584) [129] [, R.A. Rojas](https://orcid.org/0000-0002-6888-9462) [171] [, B. Roland](https://orcid.org/0000-0003-3397-6475) [50],
[C.P.A. Roland](https://orcid.org/0000-0003-2084-369X) [63] [, J. Roloff](https://orcid.org/0000-0001-6479-3079) [27] [, A. Romaniouk](https://orcid.org/0000-0001-9241-1189) [108] [, M. Romano](https://orcid.org/0000-0002-6609-7250) [21b] [, A.C. Romero Hernandez](https://orcid.org/0000-0001-9434-1380) [168],
[N. Rompotis](https://orcid.org/0000-0003-2577-1875) [88] [, M. Ronzani](https://orcid.org/0000-0002-8583-6063) [121] [, L. Roos](https://orcid.org/0000-0001-7151-9983) [131] [, S. Rosati](https://orcid.org/0000-0003-0838-5980) [70a] [, B.J. Rosser](https://orcid.org/0000-0001-7492-831X) [132] [, E. Rossi](https://orcid.org/0000-0001-5493-6486) [162] [, E. Rossi](https://orcid.org/0000-0002-2146-677X) [4],
[E. Rossi](https://orcid.org/0000-0001-9476-9854) [67a,67b] [, L.P. Rossi](https://orcid.org/0000-0003-3104-7971) [53b] [, L. Rossini](https://orcid.org/0000-0003-0424-5729) [44] [, R. Rosten](https://orcid.org/0000-0002-9095-7142) [123] [, M. Rotaru](https://orcid.org/0000-0003-4088-6275) [25b] [, B. Rottler](https://orcid.org/0000-0002-6762-2213) [50] [, D. Rousseau](https://orcid.org/0000-0001-7613-8063) [62],
[D. Rousso](https://orcid.org/0000-0003-1427-6668) [30] [, G. Rovelli](https://orcid.org/0000-0002-3430-8746) [68a,68b] [, A. Roy](https://orcid.org/0000-0002-0116-1012) [10] [, A. Rozanov](https://orcid.org/0000-0003-0504-1453) [98] [, Y. Rozen](https://orcid.org/0000-0001-6969-0634) [156] [, X. Ruan](https://orcid.org/0000-0001-5621-6677) [31f] [, A.J. Ruby](https://orcid.org/0000-0002-6978-5964) [88],
[T.A. Ruggeri](https://orcid.org/0000-0001-9941-1966) [1] [, F. R¨uhr](https://orcid.org/0000-0003-4452-620X) [50] [, A. Ruiz-Martinez](https://orcid.org/0000-0002-5742-2541) [169] [, A. Rummler](https://orcid.org/0000-0001-8945-8760) [34] [, Z. Rurikova](https://orcid.org/0000-0003-3051-9607) [50] [, N.A. Rusakovich](https://orcid.org/0000-0003-1927-5322) [77],
[H.L. Russell](https://orcid.org/0000-0003-4181-0678) [34] [, L. Rustige](https://orcid.org/0000-0002-0292-2477) [36] [, J.P. Rutherfoord](https://orcid.org/0000-0002-4682-0667) [6] [, E.M. R¨uttinger](https://orcid.org/0000-0002-6062-0952) [145] [, M. Rybar](https://orcid.org/0000-0002-6033-004X) [138] [, E.B. Rye](https://orcid.org/0000-0001-7088-1745) [129],
[A. Ryzhov](https://orcid.org/0000-0002-0623-7426) [118] [, J.A. Sabater Iglesias](https://orcid.org/0000-0003-2328-1952) [44] [, P. Sabatini](https://orcid.org/0000-0003-0159-697X) [169] [, L. Sabetta](https://orcid.org/0000-0002-0865-5891) [70a,70b] [, H.F-W. Sadrozinski](https://orcid.org/0000-0003-0019-5410) [141],
[R. Sadykov](https://orcid.org/0000-0002-9157-6819) [77] [, F. Safai Tehrani](https://orcid.org/0000-0001-7796-0120) [70a] [, B. Safarzadeh Samani](https://orcid.org/0000-0002-0338-9707) [152] [, M. Safdari](https://orcid.org/0000-0001-8323-7318) [149] [, S. Saha](https://orcid.org/0000-0001-9296-1498) [100] [, M. Sahinsoy](https://orcid.org/0000-0002-7400-7286) [111],
[A. Sahu](https://orcid.org/0000-0002-7064-0447) [177] [, M. Saimpert](https://orcid.org/0000-0002-3765-1320) [140] [, M. Saito](https://orcid.org/0000-0001-5564-0935) [159] [, T. Saito](https://orcid.org/0000-0003-2567-6392) [159], D. Salamani [34] [, G. Salamanna](https://orcid.org/0000-0002-0861-0052) [72a,72b] [, A. Salnikov](https://orcid.org/0000-0002-3623-0161) [149],
[J. Salt](https://orcid.org/0000-0003-4181-2788) [169] [, A. Salvador Salas](https://orcid.org/0000-0001-5041-5659) [12] [, D. Salvatore](https://orcid.org/0000-0002-8564-2373) [39b,39a] [, F. Salvatore](https://orcid.org/0000-0002-3709-1554) [152] [, A. Salzburger](https://orcid.org/0000-0001-6004-3510) [34] [, D. Sammel](https://orcid.org/0000-0003-4484-1410) [50],
[D. Sampsonidis](https://orcid.org/0000-0002-9571-2304) [158] [, D. Sampsonidou](https://orcid.org/0000-0003-0384-7672) [58d,58c] [, J. S´anchez](https://orcid.org/0000-0001-9913-310X) [169] [, A. Sanchez Pineda](https://orcid.org/0000-0001-8241-7835) [4] [, V. Sanchez Sebastian](https://orcid.org/0000-0002-4143-6201) [169],
[H. Sandaker](https://orcid.org/0000-0001-5235-4095) [129] [, C.O. Sander](https://orcid.org/0000-0003-2576-259X) [44] [, I.G. Sanderswood](https://orcid.org/0000-0001-7731-6757) [87] [, J.A. Sandesara](https://orcid.org/0000-0002-6016-8011) [99] [, M. Sandhoff](https://orcid.org/0000-0002-7601-8528) [177] [, C. Sandoval](https://orcid.org/0000-0003-1038-723X) [20b],
[D.P.C. Sankey](https://orcid.org/0000-0003-0955-4213) [139] [, M. Sannino](https://orcid.org/0000-0001-7700-8383) [53b,53a] [, A. Sansoni](https://orcid.org/0000-0002-9166-099X) [49] [, C. Santoni](https://orcid.org/0000-0002-1642-7186) [36] [, H. Santos](https://orcid.org/0000-0003-1710-9291) [135a,135b] [, S.N. Santpur](https://orcid.org/0000-0001-6467-9970) [16],
[A. Santra](https://orcid.org/0000-0003-4644-2579) [175] [, K.A. Saoucha](https://orcid.org/0000-0001-9150-640X) [145] [, A. Sapronov](https://orcid.org/0000-0001-7569-2548) [77] [, J.G. Saraiva](https://orcid.org/0000-0002-7006-0864) [135a,135d] [, J. Sardain](https://orcid.org/0000-0002-6932-2804) [98] [, O. Sasaki](https://orcid.org/0000-0002-2910-3906) [79] [, K. Sato](https://orcid.org/0000-0001-8988-4065) [164],
C. Sauer [59b] [, F. Sauerburger](https://orcid.org/0000-0001-8794-3228) [50] [, E. Sauvan](https://orcid.org/0000-0003-1921-2647) [4] [, P. Savard](https://orcid.org/0000-0001-5606-0107) [162,ai] [, R. Sawada](https://orcid.org/0000-0002-2226-9874) [159] [, C. Sawyer](https://orcid.org/0000-0002-2027-1428) [139] [, L. Sawyer](https://orcid.org/0000-0001-8295-0605) [93],
I. Sayago Galvan [169] [, C. Sbarra](https://orcid.org/0000-0002-8236-5251) [21b] [, A. Sbrizzi](https://orcid.org/0000-0002-1934-3041) [21b,21a] [, T. Scanlon](https://orcid.org/0000-0002-2746-525X) [92] [, J. Schaarschmidt](https://orcid.org/0000-0002-0433-6439) [144] [, P. Schacht](https://orcid.org/0000-0002-7215-7977) [111],
[D. Schaefer](https://orcid.org/0000-0002-8637-6134) [35] [, U. Sch¨afer](https://orcid.org/0000-0003-4489-9145) [96] [, A.C. Schaffer](https://orcid.org/0000-0002-2586-7554) [62] [, D. Schaile](https://orcid.org/0000-0001-7822-9663) [110] [, R.D. Schamberger](https://orcid.org/0000-0003-1218-425X) [151] [, E. Schanet](https://orcid.org/0000-0002-8719-4682) [110],
[C. Scharf](https://orcid.org/0000-0002-0294-1205) [17] [, N. Scharmberg](https://orcid.org/0000-0001-5180-3645) [97] [, V.A. Schegelsky](https://orcid.org/0000-0003-1870-1967) [133] [, D. Scheirich](https://orcid.org/0000-0001-6012-7191) [138] [, F. Schenck](https://orcid.org/0000-0001-8279-4753) [17] [, M. Schernau](https://orcid.org/0000-0002-0859-4312) [166],
[C. Schiavi](https://orcid.org/0000-0003-0957-4994) [53b,53a] [, L.K. Schildgen](https://orcid.org/0000-0002-6834-9538) [22] [, Z.M. Schillaci](https://orcid.org/0000-0002-6978-5323) [24] [, E.J. Schioppa](https://orcid.org/0000-0002-1369-9944) [65a,65b] [, M. Schioppa](https://orcid.org/0000-0003-0628-0579) [39b,39a] [, B. Schlag](https://orcid.org/0000-0002-1284-4169) [96],
[K.E. Schleicher](https://orcid.org/0000-0002-2917-7032) [50] [, S. Schlenker](https://orcid.org/0000-0001-5239-3609) [34] [, K. Schmieden](https://orcid.org/0000-0003-1978-4928) [96] [, C. Schmitt](https://orcid.org/0000-0003-1471-690X) [96] [, S. Schmitt](https://orcid.org/0000-0001-8387-1853) [44] [, L. Schoeffel](https://orcid.org/0000-0002-8081-2353) [140],
[A. Schoening](https://orcid.org/0000-0002-4499-7215) [59b] [, P.G. Scholer](https://orcid.org/0000-0003-2882-9796) [50] [, E. Schopf](https://orcid.org/0000-0002-9340-2214) [130] [, M. Schott](https://orcid.org/0000-0002-4235-7265) [96] [, J. Schovancova](https://orcid.org/0000-0003-0016-5246) [34] [, S. Schramm](https://orcid.org/0000-0001-9031-6751) [52],
[F. Schroeder](https://orcid.org/0000-0002-7289-1186) [177] [, H-C. Schultz-Coulon](https://orcid.org/0000-0002-0860-7240) [59a] [, M. Schumacher](https://orcid.org/0000-0002-1733-8388) [50] [, B.A. Schumm](https://orcid.org/0000-0002-5394-0317) [141] [, Ph. Schune](https://orcid.org/0000-0002-3971-9595) [140],
[A. Schwartzman](https://orcid.org/0000-0002-6680-8366) [149] [, T.A. Schwarz](https://orcid.org/0000-0001-5660-2690) [102] [, Ph. Schwemling](https://orcid.org/0000-0003-0989-5675) [140] [, R. Schwienhorst](https://orcid.org/0000-0001-6348-5410) [103] [, A. Sciandra](https://orcid.org/0000-0001-7163-501X) [141],
[G. Sciolla](https://orcid.org/0000-0002-8482-1775) [24] [, F. Scuri](https://orcid.org/0000-0001-9569-3089) [69a], F. Scutti [101] [, C.D. Sebastiani](https://orcid.org/0000-0003-1073-035X) [88] [, K. Sedlaczek](https://orcid.org/0000-0003-2052-2386) [45] [, P. Seema](https://orcid.org/0000-0002-3727-5636) [17] [, S.C. Seidel](https://orcid.org/0000-0002-1181-3061) [113],
[A. Seiden](https://orcid.org/0000-0003-4311-8597) [141] [, B.D. Seidlitz](https://orcid.org/0000-0002-4703-000X) [27] [, T. Seiss](https://orcid.org/0000-0003-0810-240X) [35] [, C. Seitz](https://orcid.org/0000-0003-4622-6091) [44] [, J.M. Seixas](https://orcid.org/0000-0001-5148-7363) [78b] [, G. Sekhniaidze](https://orcid.org/0000-0002-4116-5309) [67a] [, S.J. Sekula](https://orcid.org/0000-0002-3199-4699) [40],
[L. Selem](https://orcid.org/0000-0002-8739-8554) [4] [, N. Semprini-Cesari](https://orcid.org/0000-0002-3946-377X) [21b,21a] [, S. Sen](https://orcid.org/0000-0003-1240-9586) [47] [, C. Serfon](https://orcid.org/0000-0001-7658-4901) [27] [, L. Serin](https://orcid.org/0000-0003-3238-5382) [62] [, L. Serkin](https://orcid.org/0000-0003-4749-5250) [64a,64b] [, M. Sessa](https://orcid.org/0000-0002-1402-7525) [72a,72b],
[H. Severini](https://orcid.org/0000-0003-3316-846X) [124] [, S. Sevova](https://orcid.org/0000-0001-6785-1334) [149] [, F. Sforza](https://orcid.org/0000-0002-4065-7352) [53b,53a] [, A. Sfyrla](https://orcid.org/0000-0002-3003-9905) [52] [, E. Shabalina](https://orcid.org/0000-0003-4849-556X) [51] [, R. Shaheen](https://orcid.org/0000-0002-2673-8527) [150],
[J.D. Shahinian](https://orcid.org/0000-0002-1325-3432) [132] [, N.W. Shaikh](https://orcid.org/0000-0001-9358-3505) [43a,43b] [, D. Shaked Renous](https://orcid.org/0000-0002-5376-1546) [175] [, L.Y. Shan](https://orcid.org/0000-0001-9134-5925) [13a] [, M. Shapiro](https://orcid.org/0000-0001-8540-9654) [16] [, A. Sharma](https://orcid.org/0000-0002-5211-7177) [34],
[A.S. Sharma](https://orcid.org/0000-0003-2250-4181) [1] [, S. Sharma](https://orcid.org/0000-0002-0190-7558) [44] [, P.B. Shatalov](https://orcid.org/0000-0001-7530-4162) [119] [, K. Shaw](https://orcid.org/0000-0001-9182-0634) [152] [, S.M. Shaw](https://orcid.org/0000-0002-8958-7826) [97] [, P. Sherwood](https://orcid.org/0000-0002-6621-4111) [92] [, L. Shi](https://orcid.org/0000-0001-9532-5075) [92],
[C.O. Shimmin](https://orcid.org/0000-0002-2228-2251) [178] [, Y. Shimogama](https://orcid.org/0000-0003-3066-2788) [174] [, J.D. Shinner](https://orcid.org/0000-0002-3523-390X) [91] [, I.P.J. Shipsey](https://orcid.org/0000-0003-4050-6420) [130] [, S. Shirabe](https://orcid.org/0000-0002-3191-0061) [52] [, M. Shiyakova](https://orcid.org/0000-0002-4775-9669) [77],
[J. Shlomi](https://orcid.org/0000-0002-2628-3470) [175] [, M.J. Shochet](https://orcid.org/0000-0002-3017-826X) [35] [, J. Shojaii](https://orcid.org/0000-0002-9449-0412) [101] [, D.R. Shope](https://orcid.org/0000-0002-9453-9415) [150] [, S. Shrestha](https://orcid.org/0000-0001-7249-7456) [123] [, E.M. Shrif](https://orcid.org/0000-0001-8352-7227) [31f] [, M.J. Shroff](https://orcid.org/0000-0002-0456-786X) [171],
[E. Shulga](https://orcid.org/0000-0001-5099-7644) [175] [, P. Sicho](https://orcid.org/0000-0002-5428-813X) [136] [, A.M. Sickles](https://orcid.org/0000-0002-3246-0330) [168] [, E. Sideras Haddad](https://orcid.org/0000-0002-3206-395X) [31f] [, O. Sidiropoulou](https://orcid.org/0000-0002-1285-1350) [34] [, A. Sidoti](https://orcid.org/0000-0002-3277-1999) [21b],
[F. Siegert](https://orcid.org/0000-0002-2893-6412) [46] [, Dj. Sijacki](https://orcid.org/0000-0002-5809-9424) [14] [, J.M. Silva](https://orcid.org/0000-0002-5987-2984) [19] [, M.V. Silva Oliveira](https://orcid.org/0000-0003-2285-478X) [34] [, S.B. Silverstein](https://orcid.org/0000-0001-7734-7617) [43a], S. Simion [62],


66


[R. Simoniello](https://orcid.org/0000-0003-2042-6394) [34], N.D. Simpson [94] [, S. Simsek](https://orcid.org/0000-0002-9650-3846) [11b] [, P. Sinervo](https://orcid.org/0000-0002-5128-2373) [162] [, V. Sinetckii](https://orcid.org/0000-0001-5347-9308) [109] [, S. Singh](https://orcid.org/0000-0002-7710-4073) [148] [, S. Singh](https://orcid.org/0000-0001-5641-5713) [162],
[S. Sinha](https://orcid.org/0000-0002-3600-2804) [44] [, S. Sinha](https://orcid.org/0000-0002-2438-3785) [31f] [, M. Sioli](https://orcid.org/0000-0002-0912-9121) [21b,21a] [, I. Siral](https://orcid.org/0000-0003-4554-1831) [127] [, S.Yu. Sivoklokov](https://orcid.org/0000-0003-0868-8164) [109] [, J. Sj¨olin](https://orcid.org/0000-0002-5285-8995) [43a,43b] [, A. Skaf](https://orcid.org/0000-0003-3614-026X) [51],
[E. Skorda](https://orcid.org/0000-0003-3973-9382) [94] [, P. Skubic](https://orcid.org/0000-0001-6342-9283) [124] [, M. Slawinska](https://orcid.org/0000-0002-9386-9092) [82] [, K. Sliwa](https://orcid.org/0000-0002-1201-4771) [165], V. Smakhtin [175] [, B.H. Smart](https://orcid.org/0000-0002-7192-4097) [139] [, J. Smiesko](https://orcid.org/0000-0003-3725-2984) [138],
[S.Yu. Smirnov](https://orcid.org/0000-0002-6778-073X) [108] [, Y. Smirnov](https://orcid.org/0000-0002-2891-0781) [108] [, L.N. Smirnova](https://orcid.org/0000-0002-0447-2975) [109,r] [, O. Smirnova](https://orcid.org/0000-0003-2517-531X) [94] [, E.A. Smith](https://orcid.org/0000-0001-6480-6829) [35] [, H.A. Smith](https://orcid.org/0000-0003-2799-6672) [130],
[M. Smizanska](https://orcid.org/0000-0002-3777-4734) [87] [, K. Smolek](https://orcid.org/0000-0002-5996-7000) [137] [, A. Smykiewicz](https://orcid.org/0000-0001-6088-7094) [82] [, A.A. Snesarev](https://orcid.org/0000-0002-9067-8362) [107] [, H.L. Snoek](https://orcid.org/0000-0003-4579-2120) [115] [, S. Snyder](https://orcid.org/0000-0001-8610-8423) [27],
[R. Sobie](https://orcid.org/0000-0001-7430-7599) [171,y] [, A. Soffer](https://orcid.org/0000-0002-0749-2146) [157] [, F. Sohns](https://orcid.org/0000-0001-6959-2997) [51] [, C.A. Solans Sanchez](https://orcid.org/0000-0002-0518-4086) [34] [, E.Yu. Soldatov](https://orcid.org/0000-0003-0694-3272) [108] [, U. Soldevila](https://orcid.org/0000-0002-7674-7878) [169],
[A.A. Solodkov](https://orcid.org/0000-0002-2737-8674) [118] [, S. Solomon](https://orcid.org/0000-0002-7378-4454) [50] [, A. Soloshenko](https://orcid.org/0000-0001-9946-8188) [77] [, O.V. Solovyanov](https://orcid.org/0000-0002-2598-5657) [118] [, V. Solovyev](https://orcid.org/0000-0002-9402-6329) [133] [, P. Sommer](https://orcid.org/0000-0003-1703-7304) [145],
[H. Son](https://orcid.org/0000-0003-2225-9024) [165] [, A. Sonay](https://orcid.org/0000-0003-4435-4962) [12] [, W.Y. Song](https://orcid.org/0000-0003-1338-2741) [163b] [, A. Sopczak](https://orcid.org/0000-0001-6981-0544) [137], A.L. Sopio [92] [, F. Sopkova](https://orcid.org/0000-0002-6171-1119) [26b] [, S. Sottocornola](https://orcid.org/0000-0002-1430-5994) [68a,68b],
[R. Soualah](https://orcid.org/0000-0003-0124-3410) [64a,64c] [, A.M. Soukharev](https://orcid.org/0000-0002-2210-0913) [117b,117a] [, Z. Soumaimi](https://orcid.org/0000-0002-8120-478X) [33e] [, D. South](https://orcid.org/0000-0002-0786-6304) [44] [, S. Spagnolo](https://orcid.org/0000-0001-7482-6348) [65a,65b] [, M. Spalla](https://orcid.org/0000-0001-5813-1693) [111],
[M. Spangenberg](https://orcid.org/0000-0001-8265-403X) [173] [, F. Span`o](https://orcid.org/0000-0002-6551-1878) [91] [, D. Sperlich](https://orcid.org/0000-0003-4454-6999) [50] [, T.M. Spieker](https://orcid.org/0000-0002-9408-895X) [59a] [, G. Spigo](https://orcid.org/0000-0003-4183-2594) [34] [, M. Spina](https://orcid.org/0000-0002-0418-4199) [152] [, D.P. Spiteri](https://orcid.org/0000-0002-9226-2539) [55],
[M. Spousta](https://orcid.org/0000-0001-5644-9526) [138] [, A. Stabile](https://orcid.org/0000-0002-6868-8329) [66a,66b] [, B.L. Stamas](https://orcid.org/0000-0001-5430-4702) [116] [, R. Stamen](https://orcid.org/0000-0001-7282-949X) [59a] [, M. Stamenkovic](https://orcid.org/0000-0003-2251-0610) [115] [, A. Stampekis](https://orcid.org/0000-0002-7666-7544) [19],
[M. Standke](https://orcid.org/0000-0002-2610-9608) [22] [, E. Stanecka](https://orcid.org/0000-0003-2546-0516) [82] [, B. Stanislaus](https://orcid.org/0000-0001-9007-7658) [34] [, M.M. Stanitzki](https://orcid.org/0000-0002-7561-1960) [44] [, M. Stankaityte](https://orcid.org/0000-0002-2224-719X) [130] [, B. Stapf](https://orcid.org/0000-0001-5374-6402) [44],
[E.A. Starchenko](https://orcid.org/0000-0002-8495-0630) [118] [, G.H. Stark](https://orcid.org/0000-0001-6616-3433) [141] [, J. Stark](https://orcid.org/0000-0002-1217-672X) [98], D.M. Starko [163b] [, P. Staroba](https://orcid.org/0000-0001-6009-6321) [136] [, P. Starovoitov](https://orcid.org/0000-0003-1990-0992) [59a] [, S. Starz](https://orcid.org/0000-0002-2908-3909) ¨ [100],
[R. Staszewski](https://orcid.org/0000-0001-7708-9259) [82] [, G. Stavropoulos](https://orcid.org/0000-0002-8549-6855) [42] [, P. Steinberg](https://orcid.org/0000-0002-5349-8370) [27] [, A.L. Steinhebel](https://orcid.org/0000-0002-4080-2919) [127] [, B. Stelzer](https://orcid.org/0000-0003-4091-1784) [148,163a] [, H.J. Stelzer](https://orcid.org/0000-0003-0690-8573) [134],
[O. Stelzer-Chilton](https://orcid.org/0000-0002-0791-9728) [163a] [, H. Stenzel](https://orcid.org/0000-0002-4185-6484) [54] [, T.J. Stevenson](https://orcid.org/0000-0003-2399-8945) [152] [, G.A. Stewart](https://orcid.org/0000-0003-0182-7088) [34] [, M.C. Stockton](https://orcid.org/0000-0001-9679-0323) [34] [, G. Stoicea](https://orcid.org/0000-0002-7511-4614) [25b],
[M. Stolarski](https://orcid.org/0000-0003-0276-8059) [135a] [, S. Stonjek](https://orcid.org/0000-0001-7582-6227) [111] [, A. Straessner](https://orcid.org/0000-0003-2460-6659) [46] [, J. Strandberg](https://orcid.org/0000-0002-8913-0981) [150] [, S. Strandberg](https://orcid.org/0000-0001-7253-7497) [43a,43b] [, M. Strauss](https://orcid.org/0000-0002-0465-5472) [124],
[T. Strebler](https://orcid.org/0000-0002-6972-7473) [98] [, P. Strizenec](https://orcid.org/0000-0003-0958-7656) [26b] [, R. Str¨ohmer](https://orcid.org/0000-0002-0062-2438) [172] [, D.M. Strom](https://orcid.org/0000-0002-8302-386X) [127] [, L.R. Strom](https://orcid.org/0000-0002-4496-1626) [44] [, R. Stroynowski](https://orcid.org/0000-0002-7863-3778) [40],
[A. Strubig](https://orcid.org/0000-0002-2382-6951) [43a,43b] [, S.A. Stucci](https://orcid.org/0000-0002-1639-4484) [27] [, B. Stugu](https://orcid.org/0000-0002-1728-9272) [15] [, J. Stupak](https://orcid.org/0000-0001-9610-0783) [124] [, N.A. Styles](https://orcid.org/0000-0001-6976-9457) [44] [, D. Su](https://orcid.org/0000-0001-6980-0215) [149] [, S. Su](https://orcid.org/0000-0002-7356-4961) [58a] [, W. Su](https://orcid.org/0000-0001-7755-5280) [58d,144,58c],
[X. Su](https://orcid.org/0000-0001-9155-3898) [58a] [, K. Sugizaki](https://orcid.org/0000-0003-4364-006X) [159] [, V.V. Sulin](https://orcid.org/0000-0003-3943-2495) [107] [, M.J. Sullivan](https://orcid.org/0000-0002-4807-6448) [88] [, D.M.S. Sultan](https://orcid.org/0000-0003-2925-279X) [52] [, L. Sultanaliyeva](https://orcid.org/0000-0002-0059-0165) [107],
[S. Sultansoy](https://orcid.org/0000-0003-2340-748X) [3c] [, T. Sumida](https://orcid.org/0000-0002-2685-6187) [83] [, S. Sun](https://orcid.org/0000-0001-8802-7184) [102] [, S. Sun](https://orcid.org/0000-0001-5295-6563) [176] [, X. Sun](https://orcid.org/0000-0003-4409-4574) [97] [, O. Sunneborn Gudnadottir](https://orcid.org/0000-0002-6277-1877) [167], A. Suresh [48],
[C.J.E. Suster](https://orcid.org/0000-0001-7021-9380) [153] [, M.R. Sutton](https://orcid.org/0000-0003-4893-8041) [152] [, M. Svatos](https://orcid.org/0000-0002-7199-3383) [136] [, M. Swiatlowski](https://orcid.org/0000-0001-7287-0468) [163a] [, T. Swirski](https://orcid.org/0000-0002-4679-6767) [172] [, I. Sykora](https://orcid.org/0000-0003-3447-5621) [26a],
[M. Sykora](https://orcid.org/0000-0003-4422-6493) [138] [, T. Sykora](https://orcid.org/0000-0001-9585-7215) [138] [, D. Ta](https://orcid.org/0000-0002-0918-9175) [96] [, K. Tackmann](https://orcid.org/0000-0003-3917-3761) [44,w] [, A. Taffard](https://orcid.org/0000-0002-5800-4798) [166] [, R. Tafirout](https://orcid.org/0000-0003-3425-794X) [163a] [, R.H.M. Taibah](https://orcid.org/0000-0001-7002-0590) [131],
[R. Takashima](https://orcid.org/0000-0003-1466-6869) [84] [, K. Takeda](https://orcid.org/0000-0002-2611-8563) [80] [, T. Takeshita](https://orcid.org/0000-0003-1135-1423) [146] [, E.P. Takeva](https://orcid.org/0000-0003-3142-030X) [48] [, Y. Takubo](https://orcid.org/0000-0002-3143-8510) [79] [, M. Talby](https://orcid.org/0000-0001-9985-6033) [98],
[A.A. Talyshev](https://orcid.org/0000-0001-8560-3756) [117b,117a] [, K.C. Tam](https://orcid.org/0000-0002-1433-2140) [60b], N.M. Tamir [157] [, A. Tanaka](https://orcid.org/0000-0002-9166-7083) [159] [, J. Tanaka](https://orcid.org/0000-0001-9994-5802) [159] [, R. Tanaka](https://orcid.org/0000-0002-9929-1797) [62], J. Tang [58c],
[Z. Tao](https://orcid.org/0000-0003-0362-8795) [170] [, S. Tapia Araya](https://orcid.org/0000-0002-3659-7270) [76] [, S. Tapprogge](https://orcid.org/0000-0003-1251-3332) [96] [, A. Tarek Abouelfadl Mohamed](https://orcid.org/0000-0002-9252-7605) [103] [, S. Tarem](https://orcid.org/0000-0002-9296-7272) [156] [, K. Tariq](https://orcid.org/0000-0002-0584-8700) [58b],
[G. Tarna](https://orcid.org/0000-0002-5060-2208) [25b] [, G.F. Tartarelli](https://orcid.org/0000-0002-4244-502X) [66a] [, P. Tas](https://orcid.org/0000-0001-5785-7548) [138] [, M. Tasevsky](https://orcid.org/0000-0002-1535-9732) [136] [, E. Tassi](https://orcid.org/0000-0002-3335-6500) [39b,39a] [, G. Tateno](https://orcid.org/0000-0003-3348-0234) [159] [, Y. Tayalati](https://orcid.org/0000-0001-8760-7259) [33e],
[G.N. Taylor](https://orcid.org/0000-0002-1831-4871) [101] [, W. Taylor](https://orcid.org/0000-0002-6596-9125) [163b], H. Teagle [88] [, A.S. Tee](https://orcid.org/0000-0003-3587-187X) [176] [, R. Teixeira De Lima](https://orcid.org/0000-0001-5545-6513) [149] [, P. Teixeira-Dias](https://orcid.org/0000-0001-9977-3836) [91],
H. Ten Kate [34] [, J.J. Teoh](https://orcid.org/0000-0003-4803-5213) [115] [, K. Terashi](https://orcid.org/0000-0001-6520-8070) [159] [, J. Terron](https://orcid.org/0000-0003-0132-5723) [95] [, S. Terzo](https://orcid.org/0000-0003-3388-3906) [12] [, M. Testa](https://orcid.org/0000-0003-1274-8967) [49] [, R.J. Teuscher](https://orcid.org/0000-0002-8768-2272) [162,y],
[N. Themistokleous](https://orcid.org/0000-0003-1882-5572) [48] [, T. Theveneaux-Pelzer](https://orcid.org/0000-0002-9746-4172) [17], O. Thielmann [177], D.W. Thomas [91] [, J.P. Thomas](https://orcid.org/0000-0001-6965-6604) [19],
[E.A. Thompson](https://orcid.org/0000-0001-7050-8203) [44] [, P.D. Thompson](https://orcid.org/0000-0002-6239-7715) [19] [, E. Thomson](https://orcid.org/0000-0001-6031-2768) [132] [, E.J. Thorpe](https://orcid.org/0000-0003-1594-9350) [90] [, Y. Tian](https://orcid.org/0000-0001-8739-9250) [51] [, V.O. Tikhomirov](https://orcid.org/0000-0002-9634-0581) [107,af],
[Yu.A. Tikhonov](https://orcid.org/0000-0002-8023-6448) [117b,117a], S. Timoshenko [108] [, P. Tipton](https://orcid.org/0000-0002-3698-3585) [178] [, S. Tisserant](https://orcid.org/0000-0002-0294-6727) [98] [, S.H. Tlou](https://orcid.org/0000-0002-4934-1661) [31f] [, A. Tnourji](https://orcid.org/0000-0003-2674-9274) [36],
[K. Todome](https://orcid.org/0000-0003-2445-1132) [21b,21a] [, S. Todorova-Nova](https://orcid.org/0000-0003-2433-231X) [138], S. Todt [46], M. Togawa [79] [, J. Tojo](https://orcid.org/0000-0003-4666-3208) [85] [, S. Tok´ar](https://orcid.org/0000-0001-8777-0590) [26a] [, K. Tokushuku](https://orcid.org/0000-0002-8262-1577) [79],
[E. Tolley](https://orcid.org/0000-0002-1027-1213) [123] [, R. Tombs](https://orcid.org/0000-0002-1824-034X) [30] [, M. Tomoto](https://orcid.org/0000-0002-4603-2070) [79,112] [, L. Tompkins](https://orcid.org/0000-0001-8127-9653) [149] [, P. Tornambe](https://orcid.org/0000-0003-1129-9792) [99] [, E. Torrence](https://orcid.org/0000-0003-2911-8910) [127] [, H. Torres](https://orcid.org/0000-0003-0822-1206) [46],
[E. Torr´o Pastor](https://orcid.org/0000-0002-5507-7924) [169] [, M. Toscani](https://orcid.org/0000-0001-9898-480X) [28] [, C. Tosciri](https://orcid.org/0000-0001-6485-2227) [35] [, J. Toth](https://orcid.org/0000-0001-9128-6080) [98,x] [, D.R. Tovey](https://orcid.org/0000-0001-5543-6192) [145], A. Traeet [15] [, C.J. Treado](https://orcid.org/0000-0002-0902-491X) [121],
[T. Trefzger](https://orcid.org/0000-0002-9820-1729) [172] [, A. Tricoli](https://orcid.org/0000-0002-8224-6105) [27] [, I.M. Trigger](https://orcid.org/0000-0002-6127-5847) [163a] [, S. Trincaz-Duvoid](https://orcid.org/0000-0001-5913-0828) [131] [, D.A. Trischuk](https://orcid.org/0000-0001-6204-4445) [170], W. Trischuk [162],
[B. Trocm´e](https://orcid.org/0000-0001-9500-2487) [56] [, A. Trofymov](https://orcid.org/0000-0001-7688-5165) [62] [, C. Troncon](https://orcid.org/0000-0002-7997-8524) [66a] [, F. Trovato](https://orcid.org/0000-0003-1041-9131) [152] [, L. Truong](https://orcid.org/0000-0001-8249-7150) [31c] [, M. Trzebinski](https://orcid.org/0000-0002-5151-7101) [82] [, A. Trzupek](https://orcid.org/0000-0001-6938-5867) [82],
[F. Tsai](https://orcid.org/0000-0001-7878-6435) [151] [, A. Tsiamis](https://orcid.org/0000-0002-8761-4632) [158], P.V. Tsiareshka [104,ad] [, A. Tsirigotis](https://orcid.org/0000-0002-6632-0440) [158,u] [, V. Tsiskaridze](https://orcid.org/0000-0002-2119-8875) [151], E.G. Tskhadadze [155a],
[M. Tsopoulou](https://orcid.org/0000-0002-9104-2884) [158] [, Y. Tsujikawa](https://orcid.org/0000-0002-8784-5684) [83] [, I.I. Tsukerman](https://orcid.org/0000-0002-8965-6676) [119] [, V. Tsulaia](https://orcid.org/0000-0001-8157-6711) [16] [, S. Tsuno](https://orcid.org/0000-0002-2055-4364) [79], O. Tsur [156] [, D. Tsybychev](https://orcid.org/0000-0001-8212-6894) [151],
[Y. Tu](https://orcid.org/0000-0002-5865-183X) [60b] [, A. Tudorache](https://orcid.org/0000-0001-6307-1437) [25b] [, V. Tudorache](https://orcid.org/0000-0001-5384-3843) [25b] [, A.N. Tuna](https://orcid.org/0000-0002-7672-7754) [34] [, S. Turchikhin](https://orcid.org/0000-0001-6506-3123) [77] [, I. Turk Cakir](https://orcid.org/0000-0002-0726-5648) [3a], R.J. Turner [19],
[R. Turra](https://orcid.org/0000-0001-8740-796X) [66a] [, P.M. Tuts](https://orcid.org/0000-0001-6131-5725) [37] [, S. Tzamarias](https://orcid.org/0000-0002-8363-1072) [158] [, P. Tzanis](https://orcid.org/0000-0001-6828-1599) [9] [, E. Tzovara](https://orcid.org/0000-0002-0410-0055) [96], K. Uchida [159] [, F. Ukegawa](https://orcid.org/0000-0002-9813-7931) [164],
[P.A. Ulloa Poblete](https://orcid.org/0000-0002-0789-7581) [142c] [, G. Unal](https://orcid.org/0000-0001-8130-7423) [34] [, M. Unal](https://orcid.org/0000-0002-1646-0621) [10] [, A. Undrus](https://orcid.org/0000-0002-1384-286X) [27] [, G. Unel](https://orcid.org/0000-0002-3274-6531) [166] [, F.C. Ungaro](https://orcid.org/0000-0003-2005-595X) [101] [, K. Uno](https://orcid.org/0000-0002-2209-8198) [159],
[J. Urban](https://orcid.org/0000-0002-7633-8441) [26b] [, P. Urquijo](https://orcid.org/0000-0002-0887-7953) [101] [, G. Usai](https://orcid.org/0000-0001-5032-7907) [7] [, R. Ushioda](https://orcid.org/0000-0002-4241-8937) [160] [, M. Usman](https://orcid.org/0000-0003-1950-0307) [106] [, Z. Uysal](https://orcid.org/0000-0002-7110-8065) [11d] [, V. Vacek](https://orcid.org/0000-0001-9584-0392) [137] [, B. Vachon](https://orcid.org/0000-0001-8703-6978) [100],
[K.O.H. Vadla](https://orcid.org/0000-0001-6729-1584) [129] [, T. Vafeiadis](https://orcid.org/0000-0003-1492-5007) [34] [, C. Valderanis](https://orcid.org/0000-0001-9362-8451) [110] [, E. Valdes Santurio](https://orcid.org/0000-0001-9931-2896) [43a,43b] [, M. Valente](https://orcid.org/0000-0002-0486-9569) [163a],
[S. Valentinetti](https://orcid.org/0000-0003-2044-6539) [21b,21a] [, A. Valero](https://orcid.org/0000-0002-9776-5880) [169] [, R.A. Vallance](https://orcid.org/0000-0002-6782-1941) [19] [, A. Vallier](https://orcid.org/0000-0002-5496-349X) [98] [, J.A. Valls Ferrer](https://orcid.org/0000-0002-3953-3117) [169] [, T.R. Van Daalen](https://orcid.org/0000-0002-2254-125X) [144],
[P. Van Gemmeren](https://orcid.org/0000-0002-7227-4006) [5] [, S. Van Stroud](https://orcid.org/0000-0002-7969-0301) [92] [, I. Van Vulpen](https://orcid.org/0000-0001-7074-5655) [115] [, M. Vanadia](https://orcid.org/0000-0003-2684-276X) [71a,71b] [, W. Vandelli](https://orcid.org/0000-0001-6581-9410) [34],
[M. Vandenbroucke](https://orcid.org/0000-0001-9055-4020) [140] [, E.R. Vandewall](https://orcid.org/0000-0003-3453-6156) [125] [, D. Vannicola](https://orcid.org/0000-0001-6814-4674) [157] [, L. Vannoli](https://orcid.org/0000-0002-9866-6040) [53b,53a] [, R. Vari](https://orcid.org/0000-0002-2814-1337) [70a] [, E.W. Varnes](https://orcid.org/0000-0001-7820-9144) [6],
[C. Varni](https://orcid.org/0000-0001-6733-4310) [16] [, T. Varol](https://orcid.org/0000-0002-0697-5808) [154] [, D. Varouchas](https://orcid.org/0000-0002-0734-4442) [62] [, K.E. Varvell](https://orcid.org/0000-0003-1017-1295) [153] [, M.E. Vasile](https://orcid.org/0000-0001-8415-0759) [25b], L. Vaslin [36] [, G.A. Vasquez](https://orcid.org/0000-0002-3285-7004) [171],


67


[F. Vazeille](https://orcid.org/0000-0003-1631-2714) [36] [, D. Vazquez Furelos](https://orcid.org/0000-0002-5551-3546) [12] [, T. Vazquez Schroeder](https://orcid.org/0000-0002-9780-099X) [34] [, J. Veatch](https://orcid.org/0000-0003-0855-0958) [51] [, V. Vecchio](https://orcid.org/0000-0002-1351-6757) [97] [, M.J. Veen](https://orcid.org/0000-0001-5284-2451) [115],
[I. Veliscek](https://orcid.org/0000-0003-2432-3309) [130] [, L.M. Veloce](https://orcid.org/0000-0003-1827-2955) [162] [, F. Veloso](https://orcid.org/0000-0002-5956-4244) [135a,135c] [, S. Veneziano](https://orcid.org/0000-0002-2598-2659) [70a] [, A. Ventura](https://orcid.org/0000-0002-3368-3413) [65a,65b] [, A. Verbytskyi](https://orcid.org/0000-0002-3713-8033) [111],
[M. Verducci](https://orcid.org/0000-0001-8209-4757) [69a,69b] [, C. Vergis](https://orcid.org/0000-0002-3228-6715) [22] [, M. Verissimo De Araujo](https://orcid.org/0000-0001-8060-2228) [78b] [, W. Verkerke](https://orcid.org/0000-0001-5468-2025) [115] [, A.T. Vermeulen](https://orcid.org/0000-0002-8884-7112) [115],
[J.C. Vermeulen](https://orcid.org/0000-0003-4378-5736) [115] [, C. Vernieri](https://orcid.org/0000-0002-0235-1053) [149] [, P.J. Verschuuren](https://orcid.org/0000-0002-4233-7563) [91] [, M. Vessella](https://orcid.org/0000-0001-8669-9139) [99] [, M.L. Vesterbacka](https://orcid.org/0000-0002-6966-5081) [121],
[M.C. Vetterli](https://orcid.org/0000-0002-7223-2965) [148,ai] [, A. Vgenopoulos](https://orcid.org/0000-0002-7011-9432) [158] [, N. Viaux Maira](https://orcid.org/0000-0002-5102-9140) [142e] [, T. Vickey](https://orcid.org/0000-0002-1596-2611) [145] [, O.E. Vickey Boeriu](https://orcid.org/0000-0002-6497-6809) [145],
[G.H.A. Viehhauser](https://orcid.org/0000-0002-0237-292X) [130] [, L. Vigani](https://orcid.org/0000-0002-6270-9176) [59b] [, M. Villa](https://orcid.org/0000-0002-9181-8048) [21b,21a] [, M. Villaplana Perez](https://orcid.org/0000-0002-0048-4602) [169], E.M. Villhauer [48],
[E. Vilucchi](https://orcid.org/0000-0002-4839-6281) [49] [, M.G. Vincter](https://orcid.org/0000-0002-5338-8972) [32] [, G.S. Virdee](https://orcid.org/0000-0002-6779-5595) [19] [, A. Vishwakarma](https://orcid.org/0000-0001-8832-0313) [48] [, C. Vittori](https://orcid.org/0000-0001-9156-970X) [21b,21a] [, I. Vivarelli](https://orcid.org/0000-0003-0097-123X) [152],
V. Vladimirov [173] [, E. Voevodina](https://orcid.org/0000-0003-2987-3772) [111] [, M. Vogel](https://orcid.org/0000-0003-0672-6868) [177] [, P. Vokac](https://orcid.org/0000-0002-3429-4778) [137] [, J. Von Ahnen](https://orcid.org/0000-0003-4032-0079) [44] [, E. Von Toerne](https://orcid.org/0000-0001-8899-4027) [22],
[V. Vorobel](https://orcid.org/0000-0001-8757-2180) [138] [, K. Vorobev](https://orcid.org/0000-0002-7110-8516) [108] [, M. Vos](https://orcid.org/0000-0001-8474-5357) [169] [, J.H. Vossebeld](https://orcid.org/0000-0001-8178-8503) [88] [, M. Vozak](https://orcid.org/0000-0002-7561-204X) [97] [, L. Vozdecky](https://orcid.org/0000-0003-2541-4827) [90] [, N. Vranjes](https://orcid.org/0000-0001-5415-5225) [14],
[M. Vranjes Milosavljevic](https://orcid.org/0000-0003-4477-9733) [14], V. Vrba [137,*] [, M. Vreeswijk](https://orcid.org/0000-0001-8083-0001) [115] [, N.K. Vu](https://orcid.org/0000-0002-6251-1178) [98] [, R. Vuillermet](https://orcid.org/0000-0003-3208-9209) [34] [, O.V. Vujinovic](https://orcid.org/0000-0003-3473-7038) [96],
[I. Vukotic](https://orcid.org/0000-0003-0472-3516) [35] [, S. Wada](https://orcid.org/0000-0002-8600-9799) [164], C. Wagner [99] [, W. Wagner](https://orcid.org/0000-0002-9198-5911) [177] [, S. Wahdan](https://orcid.org/0000-0002-6324-8551) [177] [, H. Wahlberg](https://orcid.org/0000-0003-0616-7330) [86] [, R. Wakasa](https://orcid.org/0000-0002-8438-7753) [164],
[M. Wakida](https://orcid.org/0000-0002-5808-6228) [112] [, V.M. Walbrecht](https://orcid.org/0000-0002-7385-6139) [111] [, J. Walder](https://orcid.org/0000-0002-9039-8758) [139] [, R. Walker](https://orcid.org/0000-0001-8535-4809) [110], S.D. Walker [91] [, W. Walkowiak](https://orcid.org/0000-0002-0385-3784) [147],
[A.M. Wang](https://orcid.org/0000-0001-8972-3026) [57] [, A.Z. Wang](https://orcid.org/0000-0003-2482-711X) [176] [, C. Wang](https://orcid.org/0000-0001-9116-055X) [58a] [, C. Wang](https://orcid.org/0000-0002-8487-8480) [58c] [, H. Wang](https://orcid.org/0000-0003-3952-8139) [16] [, J. Wang](https://orcid.org/0000-0002-5246-5497) [60a] [, P. Wang](https://orcid.org/0000-0002-6730-1524) [40] [, R.-J. Wang](https://orcid.org/0000-0002-5059-8456) [96],
[R. Wang](https://orcid.org/0000-0001-9839-608X) [57] [, R. Wang](https://orcid.org/0000-0001-8530-6487) [116] [, S.M. Wang](https://orcid.org/0000-0002-5821-4875) [154], S. Wang [58b] [, T. Wang](https://orcid.org/0000-0002-1152-2221) [58a] [, W.T. Wang](https://orcid.org/0000-0002-7184-9891) [75] [, W.X. Wang](https://orcid.org/0000-0002-1444-6260) [58a] [, X. Wang](https://orcid.org/0000-0002-6229-1945) [13c],
[X. Wang](https://orcid.org/0000-0002-2411-7399) [168] [, X. Wang](https://orcid.org/0000-0001-5173-2234) [58c] [, Y. Wang](https://orcid.org/0000-0003-2693-3442) [58a] [, Z. Wang](https://orcid.org/0000-0002-0928-2070) [102] [, C. Wanotayaroj](https://orcid.org/0000-0002-8178-5705) [34] [, A. Warburton](https://orcid.org/0000-0002-2298-7315) [100] [, C.P. Ward](https://orcid.org/0000-0002-5162-533X) [30],
[R.J. Ward](https://orcid.org/0000-0001-5530-9919) [19] [, N. Warrack](https://orcid.org/0000-0002-8268-8325) [55] [, A.T. Watson](https://orcid.org/0000-0001-7052-7973) [19] [, M.F. Watson](https://orcid.org/0000-0002-9724-2684) [19] [, G. Watts](https://orcid.org/0000-0002-0753-7308) [144] [, B.M. Waugh](https://orcid.org/0000-0003-0872-8920) [92] [, A.F. Webb](https://orcid.org/0000-0002-6700-7608) [10],
[C. Weber](https://orcid.org/0000-0002-8659-5767) [27] [, M.S. Weber](https://orcid.org/0000-0002-2770-9031) [18] [, S.A. Weber](https://orcid.org/0000-0003-1710-4298) [32] [, S.M. Weber](https://orcid.org/0000-0002-2841-1616) [59a], C. Wei [58a] [, Y. Wei](https://orcid.org/0000-0001-9725-2316) [130] [, A.R. Weidberg](https://orcid.org/0000-0002-5158-307X) [130],
[J. Weingarten](https://orcid.org/0000-0003-2165-871X) [45] [, M. Weirich](https://orcid.org/0000-0002-5129-872X) [96] [, C. Weiser](https://orcid.org/0000-0002-6456-6834) [50] [, T. Wenaus](https://orcid.org/0000-0002-8678-893X) [27] [, B. Wendland](https://orcid.org/0000-0003-1623-3899) [45] [, T. Wengler](https://orcid.org/0000-0002-4375-5265) [34] [, S. Wenig](https://orcid.org/0000-0002-4770-377X) [34],
[N. Wermes](https://orcid.org/0000-0001-9971-0077) [22] [, M. Wessels](https://orcid.org/0000-0002-8192-8999) [59a] [, K. Whalen](https://orcid.org/0000-0002-9383-8763) [127] [, A.M. Wharton](https://orcid.org/0000-0002-9507-1869) [87] [, A.S. White](https://orcid.org/0000-0003-0714-1466) [57] [, A. White](https://orcid.org/0000-0001-8315-9778) [7] [, M.J. White](https://orcid.org/0000-0001-5474-4580) [1],
[D. Whiteson](https://orcid.org/0000-0002-2005-3113) [166] [, L. Wickremasinghe](https://orcid.org/0000-0002-2711-4820) [128] [, W. Wiedenmann](https://orcid.org/0000-0003-3605-3633) [176] [, C. Wiel](https://orcid.org/0000-0003-1995-9185) [46] [, M. Wielers](https://orcid.org/0000-0001-9232-4827) [139], N. Wieseotte [96],
[C. Wiglesworth](https://orcid.org/0000-0001-6219-8946) [38] [, L.A.M. Wiik-Fuchs](https://orcid.org/0000-0002-5035-8102) [50], D.J. Wilbern [124] [, H.G. Wilkens](https://orcid.org/0000-0002-8483-9502) [34] [, L.J. Wilkins](https://orcid.org/0000-0002-7092-3500) [91],
[D.M. Williams](https://orcid.org/0000-0002-5646-1856) [37], H.H. Williams [132] [, S. Williams](https://orcid.org/0000-0001-6174-401X) [30] [, S. Willocq](https://orcid.org/0000-0002-4120-1453) [99] [, P.J. Windischhofer](https://orcid.org/0000-0001-5038-1399) [130],
[I. Wingerter-Seez](https://orcid.org/0000-0001-9473-7836) [4] [, F. Winklmeier](https://orcid.org/0000-0001-8290-3200) [127] [, B.T. Winter](https://orcid.org/0000-0001-9606-7688) [50], M. Wittgen [149] [, M. Wobisch](https://orcid.org/0000-0002-0688-3380) [93] [, A. Wolf](https://orcid.org/0000-0002-4368-9202) [96],
[R. W¨olker](https://orcid.org/0000-0002-7402-369X) [130], J. Wollrath [166] [, M.W. Wolter](https://orcid.org/0000-0001-9184-2921) [82] [, H. Wolters](https://orcid.org/0000-0002-9588-1773) [135a,135c] [, V.W.S. Wong](https://orcid.org/0000-0001-5975-8164) [170] [, A.F. Wongel](https://orcid.org/0000-0002-6620-6277) [44],
[S.D. Worm](https://orcid.org/0000-0002-3865-4996) [44] [, B.K. Wosiek](https://orcid.org/0000-0003-4273-6334) [82] [, K.W. Wo´zniak](https://orcid.org/0000-0003-1171-0887) [82] [, K. Wraight](https://orcid.org/0000-0002-3298-4900) [55] [, J. Wu](https://orcid.org/0000-0002-3173-0802) [13a,13d] [, S.L. Wu](https://orcid.org/0000-0001-5866-1504) [176] [, X. Wu](https://orcid.org/0000-0001-7655-389X) [52],
[Y. Wu](https://orcid.org/0000-0002-1528-4865) [58a] [, Z. Wu](https://orcid.org/0000-0002-5392-902X) [140,58a] [, J. Wuerzinger](https://orcid.org/0000-0002-4055-218X) [130] [, T.R. Wyatt](https://orcid.org/0000-0001-9690-2997) [97] [, B.M. Wynne](https://orcid.org/0000-0001-9895-4475) [48] [, S. Xella](https://orcid.org/0000-0002-0988-1655) [38] [, L. Xia](https://orcid.org/0000-0003-3073-3662) [13c], M. Xia [13b],
[J. Xiang](https://orcid.org/0000-0002-7684-8257) [60c] [, X. Xiao](https://orcid.org/0000-0002-1344-8723) [102] [, M. Xie](https://orcid.org/0000-0001-6707-5590) [58a] [, X. Xie](https://orcid.org/0000-0001-6473-7886) [58a], I. Xiotidis [152] [, D. Xu](https://orcid.org/0000-0001-6355-2767) [13a], H. Xu [58a] [, H. Xu](https://orcid.org/0000-0001-6110-2172) [58a] [, L. Xu](https://orcid.org/0000-0001-8997-3199) [58a],
[R. Xu](https://orcid.org/0000-0002-1928-1717) [132] [, T. Xu](https://orcid.org/0000-0002-0215-6151) [58a] [, W. Xu](https://orcid.org/0000-0001-5661-1917) [102] [, Y. Xu](https://orcid.org/0000-0001-9563-4804) [13b] [, Z. Xu](https://orcid.org/0000-0001-9571-3131) [58b] [, Z. Xu](https://orcid.org/0000-0001-9602-4901) [149] [, B. Yabsley](https://orcid.org/0000-0002-2680-0474) [153] [, S. Yacoob](https://orcid.org/0000-0001-6977-3456) [31a] [, N. Yamaguchi](https://orcid.org/0000-0002-6885-282X) [85],
[Y. Yamaguchi](https://orcid.org/0000-0002-3725-4800) [160], M. Yamatani [159] [, H. Yamauchi](https://orcid.org/0000-0003-2123-5311) [164] [, T. Yamazaki](https://orcid.org/0000-0003-0411-3590) [16] [, Y. Yamazaki](https://orcid.org/0000-0003-3710-6995) [80], J. Yan [58c] [, S. Yan](https://orcid.org/0000-0002-1512-5506) [130],
[Z. Yan](https://orcid.org/0000-0002-2483-4937) [23] [, H.J. Yang](https://orcid.org/0000-0001-7367-1380) [58c,58d] [, H.T. Yang](https://orcid.org/0000-0003-3554-7113) [16] [, S. Yang](https://orcid.org/0000-0002-0204-984X) [58a] [, T. Yang](https://orcid.org/0000-0002-4996-1924) [60c] [, X. Yang](https://orcid.org/0000-0002-1452-9824) [58a] [, X. Yang](https://orcid.org/0000-0002-9201-0972) [13a] [, Y. Yang](https://orcid.org/0000-0001-8524-1855) [159],
[Z. Yang](https://orcid.org/0000-0002-7374-2334) [102,58a] [, W-M. Yao](https://orcid.org/0000-0002-3335-1988) [16] [, Y.C. Yap](https://orcid.org/0000-0001-8939-666X) [44] [, H. Ye](https://orcid.org/0000-0002-4886-9851) [13c] [, J. Ye](https://orcid.org/0000-0001-9274-707X) [40] [, S. Ye](https://orcid.org/0000-0002-7864-4282) [27] [, I. Yeletskikh](https://orcid.org/0000-0003-0586-7052) [77] [, M.R. Yexley](https://orcid.org/0000-0002-1827-9201) [87],
[P. Yin](https://orcid.org/0000-0003-2174-807X) [37] [, K. Yorita](https://orcid.org/0000-0003-1988-8401) [174] [, K. Yoshihara](https://orcid.org/0000-0002-3656-2326) [76] [, C.J.S. Young](https://orcid.org/0000-0001-5858-6639) [50] [, C. Young](https://orcid.org/0000-0003-3268-3486) [149] [, M. Yuan](https://orcid.org/0000-0002-0991-5026) [102] [, R. Yuan](https://orcid.org/0000-0002-8452-0315) [58b,i] [, X. Yue](https://orcid.org/0000-0001-6956-3205) [59a],
[M. Zaazoua](https://orcid.org/0000-0002-4105-2988) [33e] [, B. Zabinski](https://orcid.org/0000-0001-5626-0993) [82] [, G. Zacharis](https://orcid.org/0000-0002-3156-4453) [9], E. Zaid [48] [, A.M. Zaitsev](https://orcid.org/0000-0002-4961-8368) [118,ae] [, T. Zakareishvili](https://orcid.org/0000-0001-7909-4772) [155b],
[N. Zakharchuk](https://orcid.org/0000-0002-4963-8836) [32] [, S. Zambito](https://orcid.org/0000-0002-4499-2545) [34] [, D. Zanzi](https://orcid.org/0000-0002-1222-7937) [50] [, S.V. Zeißner](https://orcid.org/0000-0002-9037-2152) [45] [, C. Zeitnitz](https://orcid.org/0000-0003-2280-8636) [177] [, J.C. Zeng](https://orcid.org/0000-0002-2029-2659) [168] [, D.T. Zenger Jr](https://orcid.org/0000-0002-4867-3138) [24],
[O. Zenin](https://orcid.org/0000-0002-5447-1989) [118], T. [Zeni](https://orcid.org/0000-0001-8265-6916) [ˇ] s ˇ [26a] [, S. Zenz](https://orcid.org/0000-0002-9720-1794) [90] [, S. Zerradi](https://orcid.org/0000-0001-9101-3226) [33a] [, D. Zerwas](https://orcid.org/0000-0002-4198-3029) [62] [, B. Zhang](https://orcid.org/0000-0002-9726-6707) [13c] [, D.F. Zhang](https://orcid.org/0000-0001-7335-4983) [145] [, G. Zhang](https://orcid.org/0000-0002-5706-7180) [13b],
[J. Zhang](https://orcid.org/0000-0002-9907-838X) [5] [, K. Zhang](https://orcid.org/0000-0002-9778-9209) [13a] [, L. Zhang](https://orcid.org/0000-0002-9336-9338) [13c] [, M. Zhang](https://orcid.org/0000-0001-8659-5727) [168] [, R. Zhang](https://orcid.org/0000-0002-8265-474X) [176], S. Zhang [102] [, X. Zhang](https://orcid.org/0000-0003-4731-0754) [58c] [, X. Zhang](https://orcid.org/0000-0003-4341-1603) [58b],
[Z. Zhang](https://orcid.org/0000-0002-7853-9079) [62] [, P. Zhao](https://orcid.org/0000-0003-0054-8749) [47] [, T. Zhao](https://orcid.org/0000-0002-6427-0806) [58b] [, Y. Zhao](https://orcid.org/0000-0003-0494-6728) [141] [, Z. Zhao](https://orcid.org/0000-0001-6758-3974) [58a] [, A. Zhemchugov](https://orcid.org/0000-0002-3360-4965) [77] [, Z. Zheng](https://orcid.org/0000-0002-8323-7753) [149] [, D. Zhong](https://orcid.org/0000-0001-9377-650X) [168],
B. Zhou [102] [, C. Zhou](https://orcid.org/0000-0001-5904-7258) [176] [, H. Zhou](https://orcid.org/0000-0002-7986-9045) [6] [, N. Zhou](https://orcid.org/0000-0002-1775-2511) [58c], Y. Zhou [6] [, C.G. Zhu](https://orcid.org/0000-0001-8015-3901) [58b] [, C. Zhu](https://orcid.org/0000-0002-5918-9050) [13a,13d] [, H.L. Zhu](https://orcid.org/0000-0001-8479-1345) [58a],
[H. Zhu](https://orcid.org/0000-0001-8066-7048) [13a] [, J. Zhu](https://orcid.org/0000-0002-5278-2855) [102] [, Y. Zhu](https://orcid.org/0000-0002-7306-1053) [58a] [, X. Zhuang](https://orcid.org/0000-0003-0996-3279) [13a] [, K. Zhukov](https://orcid.org/0000-0003-2468-9634) [107] [, V. Zhulanov](https://orcid.org/0000-0002-0306-9199) [117b,117a] [, D. Zieminska](https://orcid.org/0000-0002-6311-7420) [63],
[N.I. Zimine](https://orcid.org/0000-0003-0277-4870) [77] [, S. Zimmermann](https://orcid.org/0000-0002-1529-8925) [50,*], J. Zinsser [59b] [, M. Ziolkowski](https://orcid.org/0000-0002-2891-8812) [147] [, L. Zivkovi´c](https://orcid.org/0000-0003-4236-8930) [ˇ] [14] [, A. Zoccoli](https://orcid.org/0000-0002-0993-6185) [21b,21a],
[K. Zoch](https://orcid.org/0000-0003-2138-6187) [52] [, T.G. Zorbas](https://orcid.org/0000-0003-2073-4901) [145] [, O. Zormpa](https://orcid.org/0000-0003-3177-903X) [42] [, W. Zou](https://orcid.org/0000-0002-0779-8815) [37] [, L. Zwalinski](https://orcid.org/0000-0002-9397-2313) [34] .


1 Department of Physics, University of Adelaide, Adelaide; Australia.
2 Department of Physics, University of Alberta, Edmonton AB; Canada.
3 ( _𝑎_ ) Department of Physics, Ankara University, Ankara; ( _𝑏_ ) Istanbul Aydin University, Application and
Research Center for Advanced Studies, Istanbul; [(] _[𝑐]_ [)] Division of Physics, TOBB University of Economics
and Technology, Ankara; Turkey.


68


4 LAPP, Univ. Savoie Mont Blanc, CNRS/IN2P3, Annecy ; France.
5 High Energy Physics Division, Argonne National Laboratory, Argonne IL; United States of America.
6 Department of Physics, University of Arizona, Tucson AZ; United States of America.
7 Department of Physics, University of Texas at Arlington, Arlington TX; United States of America.
8 Physics Department, National and Kapodistrian University of Athens, Athens; Greece.
9 Physics Department, National Technical University of Athens, Zografou; Greece.
10 Department of Physics, University of Texas at Austin, Austin TX; United States of America.
11 ( _𝑎_ ) Bahcesehir University, Faculty of Engineering and Natural Sciences, Istanbul; ( _𝑏_ ) Istanbul Bilgi
University, Faculty of Engineering and Natural Sciences, Istanbul; [(] _[𝑐]_ [)] Department of Physics, Bogazici
University, Istanbul; [(] _[𝑑]_ [)] Department of Physics Engineering, Gaziantep University, Gaziantep; Turkey.
12 Institut de F´ısica d’Altes Energies (IFAE), Barcelona Institute of Science and Technology, Barcelona;
Spain.
13 ( _𝑎_ ) Institute of High Energy Physics, Chinese Academy of Sciences, Beijing; ( _𝑏_ ) Physics Department,
Tsinghua University, Beijing; [(] _[𝑐]_ [)] Department of Physics, Nanjing University, Nanjing; [(] _[𝑑]_ [)] University of
Chinese Academy of Science (UCAS), Beijing; China.
14 Institute of Physics, University of Belgrade, Belgrade; Serbia.
15 Department for Physics and Technology, University of Bergen, Bergen; Norway.
16 Physics Division, Lawrence Berkeley National Laboratory and University of California, Berkeley CA;
United States of America.
17 Institut f¨ur Physik, Humboldt Universit¨at zu Berlin, Berlin; Germany.
18 Albert Einstein Center for Fundamental Physics and Laboratory for High Energy Physics, University of
Bern, Bern; Switzerland.
19 School of Physics and Astronomy, University of Birmingham, Birmingham; United Kingdom.
20 ( _𝑎_ ) Facultad de Ciencias y Centro de Investigaci´ones, Universidad Antonio Nari˜no,
Bogot´a; [(] _[𝑏]_ [)] Departamento de F´ısica, Universidad Nacional de Colombia, Bogot´a; Colombia.
21 ( _𝑎_ ) Dipartimento di Fisica e Astronomia A. Righi, Universit`a di Bologna, Bologna; ( _𝑏_ ) INFN Sezione di
Bologna; Italy.
22 Physikalisches Institut, Universit¨at Bonn, Bonn; Germany.
23 Department of Physics, Boston University, Boston MA; United States of America.
24 Department of Physics, Brandeis University, Waltham MA; United States of America.
25 ( _𝑎_ ) Transilvania University of Brasov, Brasov; ( _𝑏_ ) Horia Hulubei National Institute of Physics and Nuclear
Engineering, Bucharest; [(] _[𝑐]_ [)] Department of Physics, Alexandru Ioan Cuza University of Iasi,
Iasi; [(] _[𝑑]_ [)] National Institute for Research and Development of Isotopic and Molecular Technologies, Physics
Department, Cluj-Napoca; [(] _[𝑒]_ [)] University Politehnica Bucharest, Bucharest; [(] _[ 𝑓]_ [)] West University in Timisoara,
Timisoara; Romania.
26 ( _𝑎_ ) Faculty of Mathematics, Physics and Informatics, Comenius University, Bratislava; ( _𝑏_ ) Department of
Subnuclear Physics, Institute of Experimental Physics of the Slovak Academy of Sciences, Kosice; Slovak
Republic.
27 Physics Department, Brookhaven National Laboratory, Upton NY; United States of America.
28 Departamento de F ´ ısica (FCEN) and IFIBA, Universidad de Buenos Aires and CONICET, Buenos Aires;
Argentina.
29 California State University, CA; United States of America.
30 Cavendish Laboratory, University of Cambridge, Cambridge; United Kingdom.
31 ( _𝑎_ ) Department of Physics, University of Cape Town, Cape Town; ( _𝑏_ ) iThemba Labs, Western
Cape; [(] _[𝑐]_ [)] Department of Mechanical Engineering Science, University of Johannesburg,
Johannesburg; [(] _[𝑑]_ [)] National Institute of Physics, University of the Philippines Diliman
(Philippines); [(] _[𝑒]_ [)] University of South Africa, Department of Physics, Pretoria; [(] _[ 𝑓]_ [)] School of Physics,


69


University of the Witwatersrand, Johannesburg; South Africa.
32 Department of Physics, Carleton University, Ottawa ON; Canada.
33 ( _𝑎_ ) Faculte des Sciences Ain Chock, R ´ eseau Universitaire de Physique des Hautes Energies - Universit ´ e ´
Hassan II, Casablanca; [(] _[𝑏]_ [)] Facult´e des Sciences, Universit´e Ibn-Tofail, K´enitra; [(] _[𝑐]_ [)] Facult´e des Sciences
Semlalia, Universit´e Cadi Ayyad, LPHEA-Marrakech; [(] _[𝑑]_ [)] LPMR, Facult´e des Sciences, Universit´e
Mohamed Premier, Oujda; [(] _[𝑒]_ [)] Facult´e des sciences, Universit´e Mohammed V, Rabat; [(] _[ 𝑓]_ [)] Mohammed VI
Polytechnic University, Ben Guerir; Morocco.
34 CERN, Geneva; Switzerland.
35 Enrico Fermi Institute, University of Chicago, Chicago IL; United States of America.
36 LPC, Universit´e Clermont Auvergne, CNRS/IN2P3, Clermont-Ferrand; France.
37 Nevis Laboratory, Columbia University, Irvington NY; United States of America.
38 Niels Bohr Institute, University of Copenhagen, Copenhagen; Denmark.
39 ( _𝑎_ ) Dipartimento di Fisica, Universit`a della Calabria, Rende; ( _𝑏_ ) INFN Gruppo Collegato di Cosenza,
Laboratori Nazionali di Frascati; Italy.
40 Physics Department, Southern Methodist University, Dallas TX; United States of America.
41 Physics Department, University of Texas at Dallas, Richardson TX; United States of America.
42 National Centre for Scientific Research ”Demokritos”, Agia Paraskevi; Greece.
43 ( _𝑎_ ) Department of Physics, Stockholm University; ( _𝑏_ ) Oskar Klein Centre, Stockholm; Sweden.
44 Deutsches Elektronen-Synchrotron DESY, Hamburg and Zeuthen; Germany.
45 Fakult¨at Physik, Technische Universit¨at Dortmund, Dortmund; Germany.
46 Institut f¨ur Kern- und Teilchenphysik, Technische Universit¨at Dresden, Dresden; Germany.
47 Department of Physics, Duke University, Durham NC; United States of America.
48 SUPA - School of Physics and Astronomy, University of Edinburgh, Edinburgh; United Kingdom.
49 INFN e Laboratori Nazionali di Frascati, Frascati; Italy.
50 Physikalisches Institut, Albert-Ludwigs-Universit¨at Freiburg, Freiburg; Germany.
51 II. Physikalisches Institut, Georg-August-Universit¨at G¨ottingen, G¨ottingen; Germany.
52 D´epartement de Physique Nucl´eaire et Corpusculaire, Universit´e de Gen`eve, Gen`eve; Switzerland.
53 ( _𝑎_ ) Dipartimento di Fisica, Universit`a di Genova, Genova; ( _𝑏_ ) INFN Sezione di Genova; Italy.
54 II. Physikalisches Institut, Justus-Liebig-Universit¨at Giessen, Giessen; Germany.
55 SUPA - School of Physics and Astronomy, University of Glasgow, Glasgow; United Kingdom.
56 LPSC, Universit´e Grenoble Alpes, CNRS/IN2P3, Grenoble INP, Grenoble; France.
57 Laboratory for Particle Physics and Cosmology, Harvard University, Cambridge MA; United States of
America.
58 ( _𝑎_ ) Department of Modern Physics and State Key Laboratory of Particle Detection and Electronics,
University of Science and Technology of China, Hefei; [(] _[𝑏]_ [)] Institute of Frontier and Interdisciplinary
Science and Key Laboratory of Particle Physics and Particle Irradiation (MOE), Shandong University,
Qingdao; [(] _[𝑐]_ [)] School of Physics and Astronomy, Shanghai Jiao Tong University, Key Laboratory for Particle
Astrophysics and Cosmology (MOE), SKLPPC, Shanghai; [(] _[𝑑]_ [)] Tsung-Dao Lee Institute, Shanghai; China.
59 ( _𝑎_ ) Kirchhoff-Institut f¨ur Physik, Ruprecht-Karls-Universit¨at Heidelberg, Heidelberg; ( _𝑏_ ) Physikalisches
Institut, Ruprecht-Karls-Universit¨at Heidelberg, Heidelberg; Germany.
60 ( _𝑎_ ) Department of Physics, Chinese University of Hong Kong, Shatin, N.T., Hong Kong; ( _𝑏_ ) Department
of Physics, University of Hong Kong, Hong Kong; [(] _[𝑐]_ [)] Department of Physics and Institute for Advanced
Study, Hong Kong University of Science and Technology, Clear Water Bay, Kowloon, Hong Kong; China.
61 Department of Physics, National Tsing Hua University, Hsinchu; Taiwan.
62 IJCLab, Universit´e Paris-Saclay, CNRS/IN2P3, 91405, Orsay; France.
63 Department of Physics, Indiana University, Bloomington IN; United States of America.
64 ( _𝑎_ ) INFN Gruppo Collegato di Udine, Sezione di Trieste, Udine; ( _𝑏_ ) ICTP, Trieste; ( _𝑐_ ) Dipartimento


70


Politecnico di Ingegneria e Architettura, Universit`a di Udine, Udine; Italy.
65 ( _𝑎_ ) INFN Sezione di Lecce; ( _𝑏_ ) Dipartimento di Matematica e Fisica, Universita del Salento, Lecce; Italy. `
66 ( _𝑎_ ) INFN Sezione di Milano; ( _𝑏_ ) Dipartimento di Fisica, Universit`a di Milano, Milano; Italy.
67 ( _𝑎_ ) INFN Sezione di Napoli; ( _𝑏_ ) Dipartimento di Fisica, Universit`a di Napoli, Napoli; Italy.
68 ( _𝑎_ ) INFN Sezione di Pavia; ( _𝑏_ ) Dipartimento di Fisica, Universit`a di Pavia, Pavia; Italy.
69 ( _𝑎_ ) INFN Sezione di Pisa; ( _𝑏_ ) Dipartimento di Fisica E. Fermi, Universit`a di Pisa, Pisa; Italy.
70 ( _𝑎_ ) INFN Sezione di Roma; ( _𝑏_ ) Dipartimento di Fisica, Sapienza Universit`a di Roma, Roma; Italy.
71 ( _𝑎_ ) INFN Sezione di Roma Tor Vergata; ( _𝑏_ ) Dipartimento di Fisica, Universit`a di Roma Tor Vergata,
Roma; Italy.
72 ( _𝑎_ ) INFN Sezione di Roma Tre; ( _𝑏_ ) Dipartimento di Matematica e Fisica, Universit`a Roma Tre, Roma;
Italy.
73 ( _𝑎_ ) INFN-TIFPA; ( _𝑏_ ) Universit`a degli Studi di Trento, Trento; Italy.
74 Institut f¨ur Astro- und Teilchenphysik, Leopold-Franzens-Universit¨at, Innsbruck; Austria.
75 University of Iowa, Iowa City IA; United States of America.
76 Department of Physics and Astronomy, Iowa State University, Ames IA; United States of America.
77 Joint Institute for Nuclear Research, Dubna; Russia.
78 ( _𝑎_ ) Departamento de Engenharia El´etrica, Universidade Federal de Juiz de Fora (UFJF), Juiz de
Fora; [(] _[𝑏]_ [)] Universidade Federal do Rio De Janeiro COPPE/EE/IF, Rio de Janeiro; [(] _[𝑐]_ [)] Instituto de F´ısica,

Universidade de S˜ao Paulo, S˜ao Paulo; Brazil.
79 KEK, High Energy Accelerator Research Organization, Tsukuba; Japan.
80 Graduate School of Science, Kobe University, Kobe; Japan.
81 ( _𝑎_ ) AGH University of Science and Technology, Faculty of Physics and Applied Computer Science,
Krakow; [(] _[𝑏]_ [)] Marian Smoluchowski Institute of Physics, Jagiellonian University, Krakow; Poland.
82 Institute of Nuclear Physics Polish Academy of Sciences, Krakow; Poland.
83 Faculty of Science, Kyoto University, Kyoto; Japan.
84 Kyoto University of Education, Kyoto; Japan.
85 Research Center for Advanced Particle Physics and Department of Physics, Kyushu University, Fukuoka ;
Japan.
86 Instituto de F´ısica La Plata, Universidad Nacional de La Plata and CONICET, La Plata; Argentina.
87 Physics Department, Lancaster University, Lancaster; United Kingdom.
88 Oliver Lodge Laboratory, University of Liverpool, Liverpool; United Kingdom.
89 Department of Experimental Particle Physics, Joˇzef Stefan Institute and Department of Physics,
University of Ljubljana, Ljubljana; Slovenia.
90 School of Physics and Astronomy, Queen Mary University of London, London; United Kingdom.
91 Department of Physics, Royal Holloway University of London, Egham; United Kingdom.
92 Department of Physics and Astronomy, University College London, London; United Kingdom.
93 Louisiana Tech University, Ruston LA; United States of America.
94 Fysiska institutionen, Lunds universitet, Lund; Sweden.
95 Departamento de F´ısica Teorica C-15 and CIAFF, Universidad Aut´onoma de Madrid, Madrid; Spain.
96 Institut f¨ur Physik, Universit¨at Mainz, Mainz; Germany.
97 School of Physics and Astronomy, University of Manchester, Manchester; United Kingdom.
98 CPPM, Aix-Marseille Universit´e, CNRS/IN2P3, Marseille; France.
99 Department of Physics, University of Massachusetts, Amherst MA; United States of America.
100 Department of Physics, McGill University, Montreal QC; Canada.
101 School of Physics, University of Melbourne, Victoria; Australia.
102 Department of Physics, University of Michigan, Ann Arbor MI; United States of America.
103 Department of Physics and Astronomy, Michigan State University, East Lansing MI; United States of


71


America.
104 B.I. Stepanov Institute of Physics, National Academy of Sciences of Belarus, Minsk; Belarus.
105 Research Institute for Nuclear Problems of Byelorussian State University, Minsk; Belarus.
106 Group of Particle Physics, University of Montreal, Montreal QC; Canada.
107 P.N. Lebedev Physical Institute of the Russian Academy of Sciences, Moscow; Russia.
108 National Research Nuclear University MEPhI, Moscow; Russia.
109 D.V. Skobeltsyn Institute of Nuclear Physics, M.V. Lomonosov Moscow State University, Moscow;
Russia.
110 Fakult¨at f¨ur Physik, Ludwig-Maximilians-Universit¨at M¨unchen, M¨unchen; Germany.
111 Max-Planck-Institut f¨ur Physik (Werner-Heisenberg-Institut), M¨unchen; Germany.
112 Graduate School of Science and Kobayashi-Maskawa Institute, Nagoya University, Nagoya; Japan.
113 Department of Physics and Astronomy, University of New Mexico, Albuquerque NM; United States of
America.
114 Institute for Mathematics, Astrophysics and Particle Physics, Radboud University/Nikhef, Nijmegen;
Netherlands.
115 Nikhef National Institute for Subatomic Physics and University of Amsterdam, Amsterdam;
Netherlands.
116 Department of Physics, Northern Illinois University, DeKalb IL; United States of America.
117 ( _𝑎_ ) Budker Institute of Nuclear Physics and NSU, SB RAS, Novosibirsk; ( _𝑏_ ) Novosibirsk State University
Novosibirsk; Russia.
118 Institute for High Energy Physics of the National Research Centre Kurchatov Institute, Protvino; Russia.
119 Institute for Theoretical and Experimental Physics named by A.I. Alikhanov of National Research
Centre ”Kurchatov Institute”, Moscow; Russia.
120 ( _𝑎_ ) New York University Abu Dhabi, Abu Dhabi; ( _𝑏_ ) United Arab Emirates University, Al
Ain; [(] _[𝑐]_ [)] University of Sharjah, Sharjah; United Arab Emirates.
121 Department of Physics, New York University, New York NY; United States of America.
122 Ochanomizu University, Otsuka, Bunkyo-ku, Tokyo; Japan.
123 Ohio State University, Columbus OH; United States of America.
124 Homer L. Dodge Department of Physics and Astronomy, University of Oklahoma, Norman OK; United
States of America.
125 Department of Physics, Oklahoma State University, Stillwater OK; United States of America.
126 Palack´y University, Joint Laboratory of Optics, Olomouc; Czech Republic.
127 Institute for Fundamental Science, University of Oregon, Eugene, OR; United States of America.
128 Graduate School of Science, Osaka University, Osaka; Japan.
129 Department of Physics, University of Oslo, Oslo; Norway.
130 Department of Physics, Oxford University, Oxford; United Kingdom.
131 LPNHE, Sorbonne Universit´e, Universit´e de Paris, CNRS/IN2P3, Paris; France.
132 Department of Physics, University of Pennsylvania, Philadelphia PA; United States of America.
133 Konstantinov Nuclear Physics Institute of National Research Centre ”Kurchatov Institute”, PNPI, St.
Petersburg; Russia.
134 Department of Physics and Astronomy, University of Pittsburgh, Pittsburgh PA; United States of
America.
135 ( _𝑎_ ) Laboratorio de Instrumenta ´ c¸˜ ao e F ´ ısica Experimental de Part ´ ıculas - LIP, Lisboa; ( _𝑏_ ) Departamento de
F´ısica, Faculdade de Ciˆencias, Universidade de Lisboa, Lisboa; [(] _[𝑐]_ [)] Departamento de F´ısica, Universidade
de Coimbra, Coimbra; [(] _[𝑑]_ [)] Centro de F ´ ısica Nuclear da Universidade de Lisboa, Lisboa; [(] _[𝑒]_ [)] Departamento de
F´ısica, Universidade do Minho, Braga; [(] _[ 𝑓]_ [)] Departamento de F´ısica Te´orica y del Cosmos, Universidad de
Granada, Granada (Spain); [(] _[𝑔]_ [)] Instituto Superior T´ecnico, Universidade de Lisboa, Lisboa; Portugal.


72


136 Institute of Physics of the Czech Academy of Sciences, Prague; Czech Republic.
137 Czech Technical University in Prague, Prague; Czech Republic.
138 Charles University, Faculty of Mathematics and Physics, Prague; Czech Republic.
139 Particle Physics Department, Rutherford Appleton Laboratory, Didcot; United Kingdom.
140 IRFU, CEA, Universit´e Paris-Saclay, Gif-sur-Yvette; France.
141 Santa Cruz Institute for Particle Physics, University of California Santa Cruz, Santa Cruz CA; United
States of America.
142 ( _𝑎_ ) Departamento de F´ısica, Pontificia Universidad Cat´olica de Chile, Santiago; ( _𝑏_ ) Universidad de la
Serena, La Serena; [(] _[𝑐]_ [)] Universidad Andres Bello, Department of Physics, Santiago; [(] _[𝑑]_ [)] Instituto de Alta
Investigaci´on, Universidad de Tarapac´a, Arica; [(] _[𝑒]_ [)] Departamento de F´ısica, Universidad T´ecnica Federico
Santa Mar´ıa, Valpara´ıso; Chile.
143 Universidade Federal de S˜ao Jo˜ao del Rei (UFSJ), S˜ao Jo˜ao del Rei; Brazil.
144 Department of Physics, University of Washington, Seattle WA; United States of America.
145 Department of Physics and Astronomy, University of Sheffield, Sheffield; United Kingdom.
146 Department of Physics, Shinshu University, Nagano; Japan.
147 Department Physik, Universit¨at Siegen, Siegen; Germany.
148 Department of Physics, Simon Fraser University, Burnaby BC; Canada.
149 SLAC National Accelerator Laboratory, Stanford CA; United States of America.
150 Department of Physics, Royal Institute of Technology, Stockholm; Sweden.
151 Departments of Physics and Astronomy, Stony Brook University, Stony Brook NY; United States of
America.
152 Department of Physics and Astronomy, University of Sussex, Brighton; United Kingdom.
153 School of Physics, University of Sydney, Sydney; Australia.
154 Institute of Physics, Academia Sinica, Taipei; Taiwan.
155 ( _𝑎_ ) E. Andronikashvili Institute of Physics, Iv. Javakhishvili Tbilisi State University, Tbilisi; ( _𝑏_ ) High
Energy Physics Institute, Tbilisi State University, Tbilisi; Georgia.
156 Department of Physics, Technion, Israel Institute of Technology, Haifa; Israel.
157 Raymond and Beverly Sackler School of Physics and Astronomy, Tel Aviv University, Tel Aviv; Israel.
158 Department of Physics, Aristotle University of Thessaloniki, Thessaloniki; Greece.
159 International Center for Elementary Particle Physics and Department of Physics, University of Tokyo,
Tokyo; Japan.
160 Department of Physics, Tokyo Institute of Technology, Tokyo; Japan.
161 Tomsk State University, Tomsk; Russia.
162 Department of Physics, University of Toronto, Toronto ON; Canada.
163 ( _𝑎_ ) TRIUMF, Vancouver BC; ( _𝑏_ ) Department of Physics and Astronomy, York University, Toronto ON;
Canada.
164 Division of Physics and Tomonaga Center for the History of the Universe, Faculty of Pure and Applied
Sciences, University of Tsukuba, Tsukuba; Japan.
165 Department of Physics and Astronomy, Tufts University, Medford MA; United States of America.
166 Department of Physics and Astronomy, University of California Irvine, Irvine CA; United States of
America.
167 Department of Physics and Astronomy, University of Uppsala, Uppsala; Sweden.
168 Department of Physics, University of Illinois, Urbana IL; United States of America.
169 Instituto de F ´ ısica Corpuscular (IFIC), Centro Mixto Universidad de Valencia - CSIC, Valencia; Spain.
170 Department of Physics, University of British Columbia, Vancouver BC; Canada.
171 Department of Physics and Astronomy, University of Victoria, Victoria BC; Canada.
172 Fakult¨at f¨ur Physik und Astronomie, Julius-Maximilians-Universit¨at W¨urzburg, W¨urzburg; Germany.


73


173 Department of Physics, University of Warwick, Coventry; United Kingdom.
174 Waseda University, Tokyo; Japan.
175 Department of Particle Physics and Astrophysics, Weizmann Institute of Science, Rehovot; Israel.
176 Department of Physics, University of Wisconsin, Madison WI; United States of America.
177 Fakult¨at f¨ur Mathematik und Naturwissenschaften, Fachgruppe Physik, Bergische Universit¨at
Wuppertal, Wuppertal; Germany.
178 Department of Physics, Yale University, New Haven CT; United States of America.
_𝑎_ Also at Borough of Manhattan Community College, City University of New York, New York NY; United
States of America.
_𝑏_ Also at Bruno Kessler Foundation, Trento; Italy.
_𝑐_ Also at Center for High Energy Physics, Peking University; China.
_𝑑_ Also at Centro Studi e Ricerche Enrico Fermi; Italy.
_𝑒_ Also at CERN, Geneva; Switzerland.
_𝑓_ Also at D´epartement de Physique Nucl´eaire et Corpusculaire, Universit´e de Gen`eve, Gen`eve;
Switzerland.

_𝑔_ Also at Departament de Fisica de la Universitat Autonoma de Barcelona, Barcelona; Spain.
_ℎ_ Also at Department of Financial and Management Engineering, University of the Aegean, Chios; Greece.
_𝑖_ Also at Department of Physics and Astronomy, Michigan State University, East Lansing MI; United
States of America.
_𝑗_ Also at Department of Physics and Astronomy, University of Louisville, Louisville, KY; United States of
America.
_𝑘_ Also at Department of Physics, Ben Gurion University of the Negev, Beer Sheva; Israel.
_𝑙_ Also at Department of Physics, California State University, East Bay; United States of America.
_𝑚_ Also at Department of Physics, California State University, Fresno; United States of America.
_𝑛_ Also at Department of Physics, California State University, Sacramento; United States of America.
_𝑜_ Also at Department of Physics, King’s College London, London; United Kingdom.
_𝑝_ Also at Department of Physics, St. Petersburg State Polytechnical University, St. Petersburg; Russia.
_𝑞_ Also at Department of Physics, University of Fribourg, Fribourg; Switzerland.
_𝑟_ Also at Faculty of Physics, M.V. Lomonosov Moscow State University, Moscow; Russia.
_𝑠_ Also at Faculty of Physics, Sofia University, ’St. Kliment Ohridski’, Sofia; Bulgaria.
_𝑡_ Also at Graduate School of Science, Osaka University, Osaka; Japan.
_𝑢_ Also at Hellenic Open University, Patras; Greece.
_𝑣_ Also at Institucio Catalana de Recerca i Estudis Avancats, ICREA, Barcelona; Spain.
_𝑤_ Also at Institut f¨ur Experimentalphysik, Universit¨at Hamburg, Hamburg; Germany.
_𝑥_ Also at Institute for Particle and Nuclear Physics, Wigner Research Centre for Physics, Budapest;
Hungary.
_𝑦_ Also at Institute of Particle Physics (IPP); Canada.
_𝑧_ Also at Institute of Physics, Azerbaijan Academy of Sciences, Baku; Azerbaijan.
_𝑎𝑎_ Also at Institute of Theoretical Physics, Ilia State University, Tbilisi; Georgia.
_𝑎𝑏_ Also at Instituto de Fisica Teorica, IFT-UAM/CSIC, Madrid; Spain.
_𝑎𝑐_ Also at Istanbul University, Dept. of Physics, Istanbul; Turkey.
_𝑎𝑑_ Also at Joint Institute for Nuclear Research, Dubna; Russia.
_𝑎𝑒_ Also at Moscow Institute of Physics and Technology State University, Dolgoprudny; Russia.
_𝑎𝑓_ Also at National Research Nuclear University MEPhI, Moscow; Russia.
_𝑎𝑔_ Also at Physikalisches Institut, Albert-Ludwigs-Universit¨at Freiburg, Freiburg; Germany.
_𝑎ℎ_ Also at The City College of New York, New York NY; United States of America.
_𝑎𝑖_ Also at TRIUMF, Vancouver BC; Canada.


74


_𝑎𝑗_ Also at Universita di Napoli Parthenope, Napoli; Italy.
_𝑎𝑘_ Also at University of Chinese Academy of Sciences (UCAS), Beijing; China.
_𝑎𝑙_ Also at Yeditepe University, Physics Department, Istanbul; Turkey.
∗ Deceased


75


