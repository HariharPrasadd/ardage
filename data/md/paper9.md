1

# Reconfigurable Intelligent Surfaces: Principles and Opportunities

### Yuanwei Liu, Senior Member, IEEE, Xiao Liu, Xidong Mu, Tianwei Hou, Jiaqi Xu, Marco Di Renzo, Fellow, IEEE, and Naofal Al-Dhahir Fellow, IEEE


**Abstract**


Reconfigurable intelligent surfaces (RISs), also known as intelligent reflecting surfaces (IRSs),


or large intelligent surfaces (LISs) [1], have received significant attention for their potential to enhance


the capacity and coverage of wireless networks by smartly reconfiguring the wireless propagation


environment. Therefore, RISs are considered a promising technology for the sixth-generation (6G) of


communication networks. In this context, we provide a comprehensive overview of the state-of-the-art


on RISs, with focus on their operating principles, performance evaluation, beamforming design and


resource management, applications of machine learning to RIS-enhanced wireless networks, as well as


the integration of RISs with other emerging technologies. We describe the basic principles of RISs both


from physics and communications perspectives, based on which we present performance evaluation of


multi-antenna assisted RIS systems. In addition, we systematically survey existing designs for RIS

enhanced wireless networks encompassing performance analysis, information theory, and performance


optimization perspectives. Furthermore, we survey existing research contributions that apply machine


learning for tackling challenges in dynamic scenarios, such as random fluctuations of wireless channels


Y. Liu, X. Liu, J. Xu are with the School of Electronic Engineering and Computer Science, Queen Mary University of London,
London E1 4NS, UK. (email: yuanwei.liu@qmul.ac.uk; x.liu@qmul.ac.uk; jiaqi.xu@qmul.ac.uk).
X. Mu is with School of Artificial Intelligence and Key Laboratory of Universal Wireless Communications, Ministry of
Education, Beijing University of Posts and Telecommunications, Beijing, China (email: muxidong@bupt.edu.cn).
T. Hou is with the School of Electronic and Information Engineering, Beijing Jiaotong University, Beijing 100044, China
(email: 16111019@bjtu.edu.cn).
M. Di Renzo is with Universit´e Paris-Saclay, CNRS, CentraleSup´elec, Laboratoire des Signaux et Syst`emes, 3 Rue Joliot-Curie,
91192 Gif-sur-Yvette, France. (emails: marco.di-renzo@universite-paris-saclay.fr).
M. Di Renzo’s work was supported in part by the European Commission through the H2020 ARIADNE project under grant
agreement number 871464 and through the H2020 RISE-6G project under grant agreement number 101017011.
N. Al-Dhahir is with the Department of Electrical and Computer Engineering, The University of Texas at Dallas, Richardson,
TX 75080 USA. (email: aldhahir@utdallas.edu).


1 Without loss of generality, we use the name of RIS in the remainder of this paper.


2


and user mobility in RIS-enhanced wireless networks. Last but not least, we identify major issues


and research opportunities associated with the integration of RISs and other emerging technologies for


applications to next-generation networks.


**Index Terms**


6G, intelligent reflecting surfaces (IRSs), large intelligent surfaces (LISs), machine learning, per

formance optimization, reconfigurable intelligent surfaces (RISs), wireless networks


I. I NTRODUCTION


The unprecedented demands for high quality and ubiquitous wireless services impose enor

mous challenges to existing cellular networks. Applications like rate-centric enhanced mobile


broadband (eMBB), ultra-reliable, low latency communications (URLLC), and massive machine

type communications (mMTC) services are the targets for designing the fifth-generation (5G) of


communication systems. However, the goals of the sixth-generation (6G) of wireless communi

cation systems are expected to be transformative and revolutionary encompassing applications


like data-driven, instantaneous, ultra-massive, and ubiquitous wireless connectivity, as well as


connected intelligence [1], [2]. Therefore, new transmission technologies are needed in order


to support these new applications and services. Reconfigurable intelligent surfaces (RISs), also


called intelligent reflecting surfaces (IRSs) [3], [4] or large intelligent surfaces (LISs) [5], [6],


comprise an array of reflecting elements for reconfiguring the incident signals. Owing to their


capability of proactively modifying the wireless communication environment, RISs have become


a focal point of research in wireless communications to mitigate a wide range of challenges


encountered in diverse wireless networks [7], [8]. The advantages of RISs are listed as follows:


_•_ **Easy to deploy:** RISs are nearly-passive devices, made of electromagnetic (EM) material.


As illustrated in Fig. 1, RISs can be deployed on several structures, including but not limited


to building facades, indoor walls [9], aerial platforms, roadside billboards, highway polls,


vehicle windows, as well as pedestrians’ clothes due to their low-cost.


_•_ **Spectral efficiency enhancement:** RISs are capable of reconfiguring the wireless propaga

tion environment by compensating for the power loss over long distances. Virtual line

of-sight (LoS) links between base stations (BSs) and mobile users can be formed via


3


passively reflecting the impinging radio signals. The throughput enhancement becomes


significant when the LoS link between BSs and users is blocked by obstacles, e.g., high-rise


buildings. Due to the intelligent deployment and design of RISs, a software-defined wireless


environment may be constructed, which, in turn, provides potential enhancements of the


received signal-to-interference-plus-noise ratio (SINR).


_•_ **Environment friendly:** In contrast to conventional relaying systems, e.g., amplify-and

forward (AF) and decode-and-forward (DF) [10], RISs are capable of shaping the incoming


signal by controlling the phase shift of each reflecting element instead of employing a power


amplifier [11], [12]. Thus, deploying RISs is more energy-efficient and environment friendly


than conventional AF and DF systems.


_•_ **Compatibility:** RISs support full-duplex (FD) and full-band transmission due to the fact


that they only reflect the EM waves. Additionally, RIS-enhanced wireless networks are


compatible with the standards and hardware of existing wireless networks [13].


Due to the aforementioned attractive characteristics, RISs are recognized as an effective


solution for mitigating a wide range of challenges in commercial and civilian applications.


There have been many recent studies on RISs and their contributions focus on several application


scenarios under different assumptions. As a result, the system models proposed by these research


contributions tend to be different. Thus, there is an urgent need to categorize the existing research


contributions, which is one of the main goals of this paper.


Fig. 1 illustrates the applications of RISs in diverse wireless communication networks. In


Fig. 1(a), RIS-enhanced cellular networks are illustrated, where RISs are deployed for bypassing


the obstacles between BSs and users. Thus, the quality of service (QoS) in heterogeneous


networks and the latency performance in mobile edge computing (MEC) networks are im

proved [14], [15]. On the other hand, RISs can act as a signal reflection hub to support massive


connectivity via interference mitigation in device-to-device (D2D) communication networks [16],


or RISs can cancel undesired signals by smartly designing the passive beamforming in the


context of physical layer security (PLS) [17]. Additionally, RISs can be deployed to strengthen


the received signal power of cell-edge users and mitigating the interference from neighbor


cells [18], and the power loss over long distances can be compensated in simultaneous wireless


information and power transfer (SWIPT) networks [19]–[22]. In Fig. 1(b), RIS-assisted indoor


4





































Figure 1: RISs in wireless communication networks.


communications are illustrated, where RISs can be deployed on walls for enhancing the QoS


in some rate-hungry indoor scenarios, such as virtual reality (VR) applications. Additionally,


in order to guarantee no blind spots in the coverage area of some block-sensitive scenarios,


such as visible light communications [23] and wireless fidelity (WiFi) networks, a concatenated


virtual RIS-aided LoS link between the access points (APs) and the users can be formed with


the aid of RISs, which indicates that both the propagation links between the APs and the RISs,


as well as between the RISs and the users can be in LoS. In Fig. 1(c), RIS-enhanced unmanned


systems are illustrated. RISs can be leveraged for enhancing the performance of unmanned


aerial vehicle (UAV) enabled wireless networks [24], cellular-connected UAV networks [25],


autonomous vehicular networks, autonomous underwater vehicle (AUV) networks, and intelligent


robotic networks by fully reaping the aforementioned RIS benefits. For instance, in RIS-enhanced


5


UAV-aided wireless networks, one can adjust the phase shifts of RISs instead of controlling the


movement of the UAVs in order to form concatenated virtual LoS links between the UAVs and


the users. Therefore, the UAVs can maintain the hovering status only when the concatenated


virtual LoS links cannot be formed even with the aid of RISs, which reduces the movement


manipulations and the energy consumption of UAVs. In Fig. 1(d), RIS-enhanced Internet of


Things (IoT) networks are illustrated, where RISs are exploited for assisting intelligent wireless


sensor networks [26], intelligent agriculture, and intelligent factory [27].


There are some short magazine papers [3], [10], [28], [29], surveys and tutorials [6], [7], [30]–


[33] in the literature that introduced RISs and their variants, but the focus of these papers is


different from our work. More specifically, Wu _et al._ [3] provided an overview of the applications


of RISs as reflectors in wireless communications, and identified some challenges and future


research opportunities for implementing RIS-assisted wireless networks. Liang _et al._ [6] presented


an overview of the reflective radio technology with a particular focus on the large intelligent


surface/antennas. In [7], Di Renzo _et al._ provided a comprehensive overview of employing RISs


for realizing smart radio environments in wireless networks, where an electromagnetic-based


communication-theoretic framework for analyzing and optimizing metamaterial-based RISs is


presented and a survey of recent research contributions on RISs is given. Huang _et al._ [10]


introduced the concept of holographic multiple-input and multiple-output (MIMO) surfaces


(HMIMOS), and discussed both active and passive RISs, encompassing the hardware archi

tectures, operation modes, and applications in communications. In [28], Liaskos _et al._ presented


one kind of RIS prototype, namely the HyperSurface tile, for realizing programmable wireless


environments. Gacanin _et al._ [29] gave an overview of employing artificial intelligence (AI) tools


in RIS-assisted radio environments. Di Renzo _et al._ [30] introduced the concept of smart radio


environments empowered by RISs, and discussed recent research progresses and future poten

tial challenges. Basar _et al._ [31] reviewed recent research efforts on RIS-empowered wireless


networks, identified the differences between RISs and other technologies, and presented future


research challenges and opportunities. Gong _et al._ [32] surveyed recent research works on RIS

assisted wireless networks and discussed emerging applications and implementation challenges of


RISs. From the perspective of enhancing the communication performance, Wu _et al._ [33] gave


a tutorial on design issues in RIS-assisted wireless networks, including passive beamforming


6


optimization, channel estimation, and deployment design.


Although the aforementioned magazines/surveys/tutorials presented either general concepts or


specific aspects of RISs (e.g., from a physics-based or a communication-based perspective), the


fundamental performance limits of RISs and some potential applications in wireless networks


are not covered. The comparison between widely employed mathematical tools for performance


evaluations and optimizations in RIS-enhanced wireless networks is also not discussed. Moreover,


a detailed framework based on machine learning (ML) tools for designing RIS-enhanced wireless


networks is missing, except for a short magazine paper [29]. Motivated by all the aforementioned


considerations, this paper provides a comprehensive discussion of RIS-enhanced wireless network


principles, from physics to wireless communications, and discusses research opportunities for


exploiting RISs in diverse applications, such as unmanned systems, non-orthogonal multiple


access (NOMA), and ML. Table I illustrates the comparison of this treatise with the existing


magazines/surveys/tutorials in the context of RISs.


Against the above observations, our main contributions are as follows.














|—|Classifications|Key Contents|Wu<br>et al.<br>[3]|Liang<br>et al.<br>[6]|Di Renzo<br>et al.<br>[7]|Huang<br>et al.<br>[10]|Liaskos<br>et al.<br>[28]|Gacanin<br>et al.<br>[29]|Di Renzo<br>et al.<br>[30]|Basar<br>et al.<br>[31]|Gong<br>et al.<br>[32] [33]|Wu<br>et al.|This<br>work|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|Physics-based|Electromagnetics|Distinguishing ray-optics & wave-optics perspective|||✓||||||||✓|
|Physics-based|Electromagnetics|Macroscopic description of metasurface-based RIS|||✓||||✓|✓||||
|Physics-based|Electromagnetics|Surface equivalence theorems|||✓||||✓||||✓|
|Physics-based|Electromagnetics|Distinguishing far-ﬁeld & near-ﬁeld|||✓||||||||✓|
|Physics-based|Electromagnetics|Formulating the reﬂection coefﬁcient|✓|✓||||||✓|✓|✓|✓|
|Physics-based|Electromagnetics|Power conservation principle|||✓|||||||||
|Physics-based|RIS Modeling|RIS control mechanism|✓|✓|✓|✓|✓|✓|✓|✓|✓|✓|✓|
|Physics-based|RIS Modeling|Typical tunable functions|||✓||||✓|✓|✓||✓|
|Physics-based|Realizations|RIS hardware and prototypes||✓|✓|✓|✓||✓|✓|✓|✓|✓|
|Physics-based|Realizations|RIS synthesis methods|||✓|✓|✓||✓||||✓|
|Communication-based|Performance<br>evaluation|Path loss models||✓|✓|||||✓|✓|✓|✓|
|Communication-based|Performance<br>evaluation|Stochastic geometry|||✓||||✓|✓|✓||✓|
|Communication-based|Performance<br>evaluation|Discussion on stochastic analysis tools|||||||✓||||✓|
|Communication-based|Performance<br>evaluation|Discussion on small-scale fading channels|||||||||||✓|
|Communication-based|RIS-aided<br>communication<br>design|Information-theoretic capacity limits|||||||||||✓|
|Communication-based|RIS-aided<br>communication<br>design|Passive beamforming optimization|✓|✓|✓|✓||||✓|✓|✓|✓|
|Communication-based|RIS-aided<br>communication<br>design|Resource management||||✓|||||||✓|
|Communication-based|RIS-aided<br>communication<br>design|Comparison of employed mathematical approaches|||||||||||✓|
|Communication-based|RIS-aided<br>communication<br>design|Channel estimation|✓|✓|✓|✓||||✓|✓|✓|✓|
|Communication-based|RIS-aided<br>communication<br>design|Deployment design|✓|✓|||||✓|||✓|✓|
|Communication-based|RIS-aided<br>communication<br>design|Modulation|||✓||||✓|✓||||
|Communication-based|RIS-aided<br>communication<br>design|Localization and sensing|||✓||||✓|||||
|Communication-based|ML-empowered<br>RIS|Architecture||||||✓|||||✓|
|Communication-based|ML-empowered<br>RIS|DL|||✓|✓||✓||✓|✓||✓|
|Communication-based|ML-empowered<br>RIS|RL||||||✓|||||✓|
|Communication-based|ML-empowered<br>RIS|Comparison of RL-based algorithms|||||||||||✓|
|Communication-based|ML-empowered<br>RIS|Supervised, unsupervised, and federated learning||||||✓|||||✓|
|Communication-based|Compatibility|NOMA-RIS|||✓||||||||✓|
|Communication-based|Compatibility|PLS-RIS||✓|✓|✓||||✓|✓|✓|✓|
|Communication-based|Compatibility|SWIPT-RIS||✓|✓|✓|||||✓|✓|✓|
|Communication-based|Compatibility|UAV-RIS|||✓||||||✓|✓|✓|
|Communication-based|Compatibility|AV/CV-RIS|||||||||||✓|
|Communication-based|Compatibility|MEC-RIS|||✓||||||✓|✓|✓|
|Communication-based|Compatibility|mmWave-RIS|||✓||||✓|✓|✓|✓|✓|



Table I: Comparison of this work with available magazines/surveys/tutorials. Here, “DL” refers
to “Deep Learning”, “RL” refers to “Reinforcement Learning”, “AV” refers to “Autonomous
Vehicle”, and “CV” refers to “Connected Vehicle”.


_•_ We overview the fundamental principles that govern the operation of RISs and their inter

action with the EM signals. We also survey typical RIS functions and their corresponding


7


principles. Specifically, we focus on patch-array based implementation and compare the


ray-optics perspective with the wave-optics perspective.


_•_ We develop performance evaluation techniques for multi-antenna assisted RIS systems.


Research contributions are also summarized along with their advantages and limitations.


_•_ We investigate RISs from the information-theoretic perspective, based on which we review


the protocols and approaches for jointly designing beamforming and resource allocation


schemes with different optimization objectives. Additionally, the major open research prob

lems are discussed.


_•_ We discuss the need of amalgamating ML and RISs. After reviewing the most recent research


contributions, we propose a novel framework for optimizing RIS-enhanced intelligent wire

less networks, where big data analytic and ML are leveraged for optimizing RIS-enhanced


wireless networks.


_•_ We identify major research opportunities associated with the integration of RISs into other


emerging technologies and discuss potential solutions.


As illustrated in Fig. 2, this paper is structured as follows. Section II elaborates on the


fundamental operating principles of RIS-enhanced wireless networks. Section III focuses on


the performance evaluation of multi-antenna RIS-assisted systems and the main advantages of


using RISs in wireless networks. In Section IV, the latest research activities on the joint design


of beamforming and resource allocation are discussed. The framework of ML-empowered RIS

enhanced intelligent wireless networks is presented in Section V. Finally, Section VI investigates


the integration of RISs with other emerging technologies towards the design and optimization


of 6G wireless networks.


II. RIS: F ROM P HYSICS TO W IRELESS C OMMUNICATIONS


An RIS is a two-dimensional (2D) material structure with programmable macroscopic physical


characteristics. The most important characteristic of an RIS is that its EM wave response can be


reconfigured. In contrast to conventional wireless communication networks, the channels between


the transmitters and the receivers can be controlled in RIS-aided networks. Thus, the strength of


the desired received signal can be enhanced at the terminal devices. In this section, we introduce


8










|Section II: RIS: from physics to wireless communications Section II: RIS: from physics to wireless communications|Col2|
|---|---|
|**A.** **Different categories** **of** **RISs**|**A.** **Different categories** **of** **RISs**|
|**B.** **Understanding metasurfaces from the view of physics**|**B.** **Understanding metasurfaces from the view of physics**|
|**C.** **Analysis of RISs: ray optics perspective v.s. wave optics**<br>**perspective**|**C.** **Analysis of RISs: ray optics perspective v.s. wave optics**<br>**perspective**|
|**D.** **Achieving** **tunability: patch array RIS**|**D.** **Achieving** **tunability: patch array RIS**|
|**E.** **RIS operating principles**|F. Discussions and outlook|




|A. Channel models|Col2|
|---|---|
|**B.** **Performance analysis**|**B.** **Performance analysis**|
|**C.** **Benchmark schemes**|**D.** **Discussions and outlook**|







|Col1|Section VI: Integrating RISs with other technologies towards 6G<br>A. NOMA and RIS B. PLS and RIS<br>C. SWIPT and RIS D. UAV and RIS<br>E. Autonomous Driving/Connected Vehicles and RIS<br>F. Discussions and outlook|
|---|---|
|||


|A. NOMA and RIS|B. PLS and RIS|
|---|---|
|**C.** **SWIPT and RIS**|**D.** **UAV and RIS**|
|**E.** **Autonomous Driving/Connected Vehicles and RIS**|**E.** **Autonomous Driving/Connected Vehicles and RIS**|
|**F.** **Discussions and outlook**|**F.** **Discussions and outlook**|


Figure 2: Organization of the present paper.


the fundamental principles which govern the operation of RISs and their interaction with the


EM signals. We also survey typical RIS functions and their corresponding principles.


_A. Different Categories of RISs_


Considering their structures, RISs can be realized by using metamaterial or patch-array based


technologies. Metamaterial-based RISs are referred to as metasurfaces. Deployed at different


locations, RISs can be designed to work as reflecting/refracting surfaces between the BS and the


user or waveguide surfaces operating at the BS. Considering the tuning mechanisms, RISs can be


9


Table II: LIST OF ACRONYMS


AF Amplify-and-Forward
AO Alternating Optimization
AUV Autonomous Underwater Vehicle

BC Broadcast Channel

BS Base Station

CSI Channel State Information

D2D Device-to-Device

DF Decode-and-Forward
DL Deep Learning
EE Energy Efficiency
EM Electromagnetic
FD Full-Duplex
HD Half-Duplex
IRS Intelligent Reflecting Surface
IoT Internet of Things
LIS Large Intelligent Surface
LoS Line-of-Sight
MEC Mobile Edge Computing
MIMO Multiple-Input and Multiple-Output
MISO Multiple-Input and Single-Output
ML Machine Learning
MOS Mean Opinion Score
NP Non-deterministic Polynomial-time
NOMA Non-Orthogonal Multiple Access
OFDM Orthogonal Frequency Division Multiplexing
OMA Orthogonal Multiple Access
PDF Probability Density Function
PLS Physical Layer Security
QoS Quality of Service
RIS Reconfigurable Intelligent Surface
RL Reinforcement Learning
SCA Successive Convex Approximation
SE Spectral Efficiency
SG Stochastic Geometry
SIC Successive Interference Cancelation
SIMO Single-Input and Multiple-Output
SINR Signal-to-Interference-plus-Noise Ratio
SISO Single-Input and Single-Output
SNR Signal-Noise Ratio
SWIPT Simultaneous Wireless Information and Power Transfer

UAV Unmanned Aerial Vehicle
VLC Visible Light Communication
VR Virtual Reality
WiFi Wireless Fidelity
5G Fifth-Generation

6G Sixth-Generation


reconfigured electrically, mechanically, or thermally. Depending on their energy consumptions,


RISs can be categorized as passive-lossy, passive-lossless, or active. The active or passive nature


of RISs determines their ultimate performance capabilities. It is worth mentioning, however, that


RISs cannot be completely passive because of their inherent property of being configurable.


Here, we discuss three important RIS working operations: waveguide [34], refraction [35], and


reflection [36]. With the aid of Love’s field equivalence principle [37], the reflected and refracted


EM field can be studied by introducing equivalent surface electric and magnetic currents [35].


10



**Structure**


**Power Source**


**Energy consumption**


**Tuning Mechanisms**













Figure 3: Different types of metasurfaces.


In the three working conditions, the RIS converts and radiates a wave (either induced by an


incident wave or fed by a waveguide) into a desired propagating wave in free space. The


surface equivalence principles (SEPs) including Love’s field equivalence principle and Huygens’



principle are introduced in Section II-B.


_1) Waveguide RIS:_ R. Smith _et al._ [34] presented a theoretical study of waveguide-fed


metasurfaces. The elements in the metasurface are modeled as uncoupled magnetic dipoles.


The magnitude of each dipole element is proportional to the product of the reference wave


and each element’s polarizability. By tuning the polarizability, the metasurface antenna can


perform beamforming. Each element on the metasurface serves as a micro-antenna. Compared


to conventional antenna arrays, the compact waveguide metasurface occupies less space and can


transmit towards wider angles.


_2) Refracting RIS:_ Viktar S. _et al._ [38] proposed a theoretical design of a perfectly refracting


and reflecting metasurfaces. The authors used an equivalent impedance matrix model so that


the tangential field components at the two sides of the metasurface are appropriately optimized.


Moreover, three possible device realizations are discussed: self-oscillating teleportation metasur

faces, non-local metasurfaces, and metasurfaces formed by only lossless components. The role of


omega-type bianisotropy in the design of lossless-component realizations of perfectly refractive


11


surfaces is discussed.


_3) Reflecting RIS:_ Dai _et al._ [36] designed a digital coding reflective metasurface. The


elements in the metasurface contain varactor diodes with a tunable biasing voltage. By pre

designing several digitized biasing voltage levels, each element can apply discrete phase shifts


and achieve beamforming for the reflected wave.


The rest of this section is focused on the operating principles for RISs that operate as reflectors.


_B. Understanding Metasurfaces From the View of Physics_


A wireless signal is essentially an EM wave propagating in a three-dimensional space. Atten

uation or reduction of the signal strength occurs as the EM wave propagates through the space


and interacts with the scattering objects. From basic principles of electromagnetism, the signal


power per unit area is proportional to the square of the electric field of the corresponding wave


in a given media. As far as reflective and refractive smart surfaces are concerned, this requires


the understanding of how the EM waves interact with the surrounding objects. The equivalence


principle, especially the SEP, is the building block for studying the EM wave transformations.


Some authors also call it Love’s field equivalence principle. The principle can be adopted for both


external problems (source-free region) and internal problems. Love’s field equivalence principle


states that the EM field outside or inside a close surface can be uniquely determined by the


electric and magnetic currents on the surface. As shown in Fig. 4, the equivalent problem for


the region I can be reformulated by placing equivalent currents on S that satisfy the boundary


condition for each particular case and filling the region II with the same medium of constitutive


parameters _ϵ_ and _µ_ . Thus, the equivalent currents ( _−J_ _s_, _−M_ _s_ ), together with the original source


currents ( _J_ 1, _M_ 1 ), radiate the correct fields in region I. The equivalent problem for region II can


be formulated similarly.


Love’s field equivalence principle is the theoretical foundation for analyzing the radiation


pattern of RISs. However, the SEP does not specify how to calculate the EM field produced by


the surface currents. To obtain the signal strength at an arbitrary field point, the Huygens-Fresnel


principle can be employed. The Huygens-Fresnel principle is a method of analysis applied to


problems of wave propagation, which states that every point on a wavefront is itself the source


of spherical wavelets, and the secondary wavelets emanating from different points mutually


12


interfere. The sum of these spherical wavelets forms the wavefront. Based on the Huygens

Fresnel principle, the EM field scattered by an RIS (in reflection and refraction) can be quantified


analytically.


As far as waveguide-based RISs are concerned, the operating principle can be summarized as


follows. In [39], the EM wave manipulation of the waveguide metasurface performs the coupling


between three-dimensional free space waves and two-dimensional surface waves. As a result,


the metasurface can be regarded as a hologram, which carries additional information about its


radiated signal propagating in the 3D space. After being excited by the source, this pre-designed


information is coupled into the radiated field. Fig. 5 conceptually represents a pre-designed


holographic waveguide-based RIS.


_C. Analysis of RISs: Ray Optics Perspective v.s. Wave Optics Perspective_


To characterize the interaction between an RIS and the impinging EM waves, one can adopt


approximations and tools either from the perspective of ray-optics or wave-optics. These two


perspectives have been used by physicists for a long time. Even though they are based on some


approximations, these two methods of analysis are useful in order to obtain important insights


into the interaction of light or radio waves with materials. In the RIS literature, both methods


of analysis are often employed. However, the assumptions behind their use and their physical


interpretations are intrinsically different. To shed light on their differences and similarities,


we compare the two methods in this subsection. As shown in Fig 6(a), from the ray-optics


perspective, an EM wave is modeled as a collection of geometrical rays with varying phases.


The phase of each ray increases linearly with the optical path length as its traverses through


the vacuum or other media. As a result, at each location of the _i_ -th ray, a phase (denoted


by _φ_ _i_ ) can be defined. When the ray interacts with a material, the phenomenon is studied by


determining the relationship between the change of the phase and the material refraction index.


The desirable reflected wave is obtained if the ensemble of rays obey the proper co-phase


condition. The wave-optics perspective is shown in Fig 6(b), where an EM wave is represented


by the corresponding electric field and magnetic field. At each position, each of these two


vector fields can be characterized by a time-varying complex-valued vector, with a direction, an


amplitude, and a phase. From the wave-optics perspective, the interaction between the wave and


13



M 1



E 1, H 1 Null Field


J 1

-J s,-M s



M 1



Original Fields



J 1



_Ɛ, μ_ _0_ _Ɛ_ _0_ _, μ_ _0_ _Ɛ, μ_ **S** _Ɛ, μ_



**S**



**S**



Region� Region�



Region� Region�



(a) Original problem (b) Equivalent problem for Region�


Figure 4: Love’s equivalence internal problem (for region I).







Figure 5: Conceptual illustration of the holographic impedance smart surface.


the material can be studied using the equivalent principles discussed in the previous subsection.


The points with equal phase values form a series of surfaces in space, which we refer to as


wavefronts. As a result, the desired scattered waves (reflected or refracted) are obtained if proper


wavefront transformations are performed by the RIS.


_1) Comparing Ray-Optics and Wave-Optics Perspectives:_ Table III highlights some of the


differences between the two methods of analysis. Compared with wave-optics, the ray-optics


perspective is a stronger simplification of the real system. As a result, it is easier to be adopted

|Col1|Ray-optics|Wave-optics|
|---|---|---|
|Wave representation|Geometrical rays|Vector ﬁelds|
|Theoretical foundation|Snell’s law|Maxwell’s equations|
|Surface proﬁle|Phase discontinuity|Surface impedance|
|Requirement of the reﬂected wave|Co-phase condition|Proper wavefront|
|Power ﬂow|Not accurate|Accurate|



Table III: Comparing different wave representation perspectives


14


and can produce a quick prediction about the RIS design. However, ray-optics methods fail


when considering the RIS power flow. Wave-optics methods, on the other hand, predict the


power flow by using the Poynting vector, which enables us to study the local and overall RIS


power consumption. This is an important issue to consider when designing and manufacturing


RISs. For example, the authors of [38] and [40] point out that it is impossible to realize lossless


plane-wave beam steering with locally passive RISs. One has to adopt the wave-optics perspective


to study the power flow of the system. Moreover, the authors of [40] state that, according to their


simulation results, one can expect increasingly improved performance if the RISs are designed


based on wave-optics approximation in comparison to those designed based on the ray-optics


approximation. In conclusion, both perspectives have their advantages and limitations. However,


adopting the wave-optics perspective is the most appropriate choice for most cases. A study of


the differences, in terms of power flow and reflection coefficient, between ray optics and wave


optics methods of analysis is reported in [7].


_D. Achieving Tunability: Patch Array RIS_


The EM characteristics of an RIS, such as the phase discontinuity, can be reconfigured by


tuning the surface impedance, through various mechanisms. Apart from electrical voltage, other


mechanisms can be applied, including thermal excitation, optical pump, and physical stretching.


Among them, electrical control is the most convenient choice, since the electrical voltage is easier


to be quantized and controlled by field-programmable gate array (FPGA) chips. The choice of


RIS materials include semiconductors [41] and graphene [42].


Regardless of the tuning mechanisms, we focus our attention on patch-array smart surfaces in


the following text. The general geometry layout of this type of RIS can be modeled as a periodic


(or quasi-periodic in the most general case) collection of unit cells integrated on a substrate.


For ease of description, we limit our discussion to RISs that are based on a local design, in


which the cells do not interact with each other. A local design usually results in the design of


sub-optimal RISs. A comprehensive discussion about local and non-local designs can be found in


[7]. To characterize the tunability of the RIS, the method of equivalent lumped-element circuits


can be adopted. As shown in Fig. 7, the unit cell is equivalent to a lumped-element circuit with


a load impedance _Z_ _l_ . Particularly, the equivalent load impedance can be tuned by changing the


15



















(a) Ray-optics design (b) Wave-optics design


Figure 6: Comparison between ray-optics and wave-optics perspective.


bias voltage of the varactor diode. When modeling patch-array RISs in wireless communication


systems, we can characterize, under a local design, each of its unit cells through an equivalent


reflection coefficient. For example, the reflection coefficient of the _i_ -th cell can be modeled as


follows:


_r_ _i_ = _β_ _i_ _· e_ _[jφ]_ _[i]_ (1)


where _β_ _i_ and _φ_ _i_ correspond to the amplitude response and the phase response, respectively.


As shown in [43], the equivalent reflection coefficient depends on the tuning impedance of the


lumped circuit that controls each unit cell, as well as the self and mutual impedances (if mutual


coupling cannot be ignored) at the ports of the RIS. In particular, as shown in [44] and [43],


_β_ _i_ and _φ_ _i_ in (1) are usually not completely independent with each other, i.e., _β_ _i_ = _f_ ( _φ_ _i_ ). In the


following sections, Φ( _⃗r_ _x_ ) refers to the phase discontinuity introduced by the RIS as a function


of the position on the RIS, and _φ_ _mn_ refers to the phase discontinuity of the ( _m, n_ )-th element


of a patch-array RIS.


Existing designs of patch-array RISs can apply discrete phase control and, in some cases,


amplitude control. Arun _et al._ [45] designed _RFocus_, which is a two-dimensional surface with a


rectangular array of passive antennas. The size of each passive unit is _λ/_ 4 _× λ/_ 10 and the


EM waves are either reflected or refracted. The authors show that the _RFocus_ surface can


be manufactured at a low cost, and that it can improve the median signal strength by 9 _._ 5


times. Welkie _et al._ [46] developed a low-cost device embedded in the walls of a building to


passively reflect or actively transmit radio signals. Dunna _et al._ [47] realized _ScatterMIMO_,


which uses a smart surface to increase the scattering in the environment. In their hardware


16


design, each reflector unit uses a patch antenna connected to four open-ended transmission


lines. The transmission lines provide 0, _π/_ 2, _π_ or 3 _π/_ 2 phase shifts. Based on measurements, it


is shown that _ScatterMIMO_ increases the throughput by factor of two and the signal-noise ratio


(SNR) by 4.5 dB as compared with baselines schemes. It is worth mentioning that tunability


can be achieved with implementations other than patch-array based surfaces. For example,


PIVOTAL COMMWARE proposed a new technique called _Holographic Beam Forming (HBF)_ .


The proposed holographic beamformer has a low cost and power consumption, as compared


with other transmission technologies, such as massive MIMO and phased arrays.


_E. RIS Operating Principles_


Considering single-beam reflection, a patch-array based RIS can be configured to serve a


terminal device in the far-field and near-field regions. Among the many operating functions and


configurations of RISs, _anomalous reflection_ and _beamforming_ are widely used in the context of


wireless communications. Adopting the wave-optics perspective, anomalous reflection is a wave

front transformation from a plane wave to another plane wave, while beamforming is a wavefront


transformation from a plane wave to a desired wavefront. Adopting the ray-optics perspective,


we present the operating principles of these two configurations in the following text. As far as


anomalous reflection is concerned, the RIS is designed to reflect an incident beam to a far-field


terminal, following the generalized laws of reflection [48]. As far as beamforming (also called


focusing) is concerned, the incident wave is focused towards a targeted region, often referred


to as the _focal point_ . The required RIS configuration follows the co-phase condition [49]. The


relation between these two operating principles is discussed in detail in [50]. Before presenting


these two different principles, the physical distinction between the near-field region and the


far-field region is clarified.


_1) Near Field v.s. Far Field:_ In the spirit of dimensional analysis, the characteristics of a


system can be represented by dimensionless numbers. In order to separate the near-field region


from the far-field region, a proper dimensionless number is needed. Let _L_ and _R_ _F_ denote the


antenna aperture size of the RIS and the focal distance, respectively. Assume that _z_ is the distance


of a particular field point to the RIS. Theoretically, the far-field and near-field regimes can be


differentiated as follows: The distance of 2 _L_ [2] _/λ_ is a commonly used criterion to decide the


17









Figure 7: Schemetic diagram of the varactor RIS.


boundary between the near-field and far-field regions (see [51], equation (1)). The position


corresponding to _z_ = 2 _L_ [2] _/λ_ is the boundary between the near-field region and the far-field


region. This result comes from the inspection of the power density variation with the distance


between a field point and the RIS. Within the near field where _z <_ 2 _L_ [2] _/λ_, the power density


shows significant variations. The peak position of the power density in the near-field region,


namely _R_ _F_, changes with different RIS configurations. Using proper co-phase conditions, beam


focusing can be achieved within the near-field of the RIS. It is worth mentioning that, in general,


the boundary between the near-field and far-field regimes depends on the specific configuration


of the RIS, as it was recently remarked in [50].


In general, the essential difference between the near-field region and the far-field region is


how the power density changes with distance. Consider, for example, that the RIS focuses the


wave within an area _a_ . The total energy incident on the RIS is proportional to the solid angle,


Ω spanned by the surface area of the RIS with respect to the location of the transmitter. After


the reflection, the transmitted energy is spread over the area _a_ . Thus, the power density around


the focal point is proportional to Ω _/a_ . Moreover, according to [45], the area _a_ is proportional


to _λ_ [2] (1 + 4 _z_ [2] _/L_ [2] ), as a result of the Abbe diffraction limit. In the far-field region, the second


term inside the brackets dominates and Ω _/a_ is proportional to _L_ [2] Ω _/z_ [2], which is the typical


18


spherical dissipation of the signal power with the distance. In the near-field region, the first term


dominates and the area _a_ becomes very small. As a result, a high focusing gain can be achieved.


With the aid of an RIS, in general, the signal can be enhanced in both the near-field and far

field regimes. However, the rationale of this enhancement is different. For near-field applications,


the RIS is expected to enhance the signal strength for users located at targeted locations with


respect to the RIS, while reducing the signal at other locations. For far-field applications, the


RIS is typically expected to enhance the signal strength for users located at targeted angles with


respect to the RIS.


In the following text, adopting the ray-optics perspective, we discuss the generalized laws of


refraction and reflection, as well as the corresponding co-phase condition.


_2) The Generalized Laws of Refraction and Reflection:_ From a geometrical optics perspective,


anomalous reflection and refraction from an RIS can be described by using the generalized laws


of refraction and reflection [48], which is a natural derivation of both Fermat’s principle and the


boundary conditions governed by Maxwell’s equations.


**Principle 1.** _Achieving anomalous reflection (Fig. 8(a))_


_Suppose that the phase discontinuity at the boundary is a function of the position along the_


_x direction_ Φ( _⃗r_ _x_ ) _, where ⃗r_ _x_ _is the position vector on the boundary. Moreover, suppose that the_


_derivative of the phase discontinuity exists. Then, the angle of reflection (θ_ 1 _) and the angle of_


_refraction (θ_ 2 _) are [52]:_



_θ_ 1 = _sin_ _[−]_ [1]
�



_λ_
_sinθ_ _i_ +
2 _πn_ 1



_d_ Φ


_dx_

�



_,_ (2)



_n_ 1 _λ_
_θ_ 2 = _sin_ _[−]_ [1] _sinθ_ _i_ +
_n_ 2 2 _πn_ 2
�



_d_ Φ


_dx_



�



_,_ (3)



_where θ_ _i_ _is the angle of incidence, λ is the wavelength of the transmitted signal in vacuum, and_


_n_ 1 _, n_ 2 _are the refractive indexes, as shown in Fig. 9._ 

There are other results related to the generalized laws of refraction and reflection, including the


critical angles for total internal reflection or refraction. The main result just presented here states


that, when a phase discontinuity is introduced at the boundary surface, the angles of reflection


19





User


(b) Beamforming
(beam focusing)



User A



User B User C



(a) Anomalous reflection
(beam steering)



Figure 8: Typical functions of reflecting surfaces.

















Figure 9: Illustration of the generalized laws of refraction and reflection.


and refraction depend not only on the angle of incidence but also on the wavelength, refractive


indexes, and the gradient of the phase discontinuity. This gives extra controllable parameters


to manipulate the reflected and refracted EM waves. As a result, anomalous reflection can be


achieved by tuning the phase gradient ( _d_ Φ _/dx_ ) based on (2) or, in the discrete patch-array


implementation, by tuning the length of the super-lattice. However, the assumption that the


derivative of the phase discontinuity is constant ( _d_ Φ _/dx_ = const.) does not necessary hold if


different wave transformations are needed.


_3) Co-phase condition:_ Focusing is usually implemented when the RIS is within the near-field


of the source or the terminal is close to the RIS. In these cases, the curvatures of the incident


20


and reflected wavefront are non-negligible. The optimization of the surface aims to produce a


pencil-beam pointing towards the direction of the terminal. When the link between the source


and the RIS, as well as the link between the RIS and the terminal are in LoS, the following


co-phase condition [49] can be applied.


**Principle 2.** _Achieving beamforming (focusing) (Fig. 8(b))_


_Let r_ _mn_ _denote the position of the_ ( _m, n_ ) _-th RIS element, r_ _s_ _denote the position of the source,_


_and_ ˆ _u denote the direction of the observer with respect to the surface. As shown in Fig. 10, φ_ _mn_


_can be chosen as follows [53]:_


_−_ _k_ 0 ( _|⃗r_ _mn_ _−_ _⃗r_ _s_ _| −_ _⃗r_ _mn_ _·_ ˆ _u_ ) + _φ_ _mn_ = 2 _π · t_ (4)


_where t_ = 1 _,_ 2 _,_ 3 _... and k_ 0 = 2 _π/λ_ _c_ _._ 

The two design principles just discussed provide guidance on how to configure the RIS phase


shift patterns for typical applications. In more complex wireless communication systems, the


RIS role is more intricate, thus cannot be categorized by the two working functions described


in Fig. 8. In these cases, to determine the RIS configuration, an optimization problem needs be


formulated. These issues are elaborated in the following sections.


_F. Discussions and Outlook_


In the design and configuration of RISs, both theoretical limitations and hardware implementa

tion limitations affect the overall performance of the resulting system. Theoretical limitations are,


e.g., the result of considering simplified assumptions or adopting naive perspectives during the


modeling of the RIS and its interaction with the EM wave. Hardware implementation limitations


come, e.g., from the discretization of an ideally continuous RIS profile. In the following text,


we elaborate on three major points.


_1) Hardware limitations:_ In practical application scenarios, many hardware parameters signif

icantly affect the achievable performance of the system. For example, the number of quantization


levels of the RIS phase shifts, the maximum number of elements that is possible to integrate


on the substrate, and the percentage of scattering environment that can be coated by an RIS.


21







center of

coordinate







observation

direction


Figure 10: Coordinate representation of co-phase condition.


Existing research contributions studied the limitations and tradeoffs caused by these hardware


constraints by analyzing their effect on the channel distribution [54], the power scaling law [55],


and performance metrics such as the outage probability [56] and ergodic capacity [57]. At the


time of writing, the analysis of the impact of hardware limitations and how an RIS performs


compared to other available technologies are open research issues.


_2) System design simplifications:_ The adoption of oversimplified models for RIS hardware


or channel models result in limitations on the system design. Because of the complex nature


of RISs and their interaction with the environment, initial research contributions have adopted


simple models. For example, using hardware models based on local designs, the ray-optics


perspective for channel modeling, and decoupling reflection amplitude and phase shift of the


RIS elements. Even though the ray-optics approximation can yield effective designs in some


cases (as shown in Section II-E), it is preferable to adopt the wave-optics design in practical


cases [50]. Further research efforts are needed to bridge complex physical models of different


RIS implementations with widely used communication models [43].


_3) Optimization limitations:_ To reap the benefits of deploying RISs in wireless networks, the


RIS parameters (e.g., the reflection coefficient and deployment location) and network resource al

22


location (e.g., transmit beamforming and user scheduling) should be jointly optimized. However,


the resulting joint optimization problems are normally non-convex and involve highly-coupled


variables, which make it challenging to derive a globally optimal solution. Though some efficient


algorithms have been recently proposed to compute high-quality suboptimal solutions [58]–[60],


the performance limits of RISs remain unknown. To fully understand the attainable performance


limits, information-theoretic perspective investigations [61], [62] are important and sophisticated


mathematical tools [63] are expected to be employed. Further details are given in Sections VI


and V.


III. P ERFORMANCE A NALYSIS OF M ULTI -A NTENNA A SSISTED RIS S YSTEMS


In Section II, we have discussed the fundamental physical properties of the RISs. However,


how RISs affect the communication performance is still an open problem. To systematically


survey existing designs for RIS-enhanced networks, we discuss the following topics: (1) channel


models, (2) performance analysis, and (3) benchmark schemes.


_A._ _**Channel Models**_


**1) Path Loss Models**


Some research contributions on the path loss for RISs are available in [7], [64] and [50], which


showed that the power scattered by an RIS is usually formulated in terms of an integral that


accounts, by leveraging the Huygens principle, for the impact of the entire surface in the free

space scenario, where scattering, shadowing, and reflection are ignored. Closed-form expressions


of the integral are, on the other hand, difficult to obtain, except for some asymptotic regimes,


which correspond to viewing the RIS as electrically small and electrically large (with respect


to the wavelength and the transmission distances). It is worth mentioning, in addition, that the


path-loss model depends on the particular phase gradient applied by the RIS. Notably, the scaling


laws can be different if the RIS operates as an anomalous reflector and as a focusing lens. In


the following, we briefly discuss two scaling laws that have recently been reported for RISs


that operate as anomalous reflectors (described in the previous section). Further information and


details can be found in [7] and [50].


23


_•_ **Electrically Small RISs:** In this asymptotic regime, the RIS is assumed to be relatively


small in size compared with the transmission distances. In this regime, the RIS can be


approximated as a small-size scatterer. In general, the path-loss scales with the reciprocal


of the product of the distance between the transmitter and the center of the RIS and the


distance between the center of the RIS and the receiver. In addition, the received power


usually increases with the size of the RIS. The received power is usually maximized in the


direction of anomalous reflection, where the path loss through the RIS follows the “product


of distances” models, which can be formulated as:


_L_ ( _d_ _SR_ _, d_ _RU_ ) _≈_ _λ_ _S_ ( _d_ _SR_ _d_ _RU_ ) _[−]_ [1] _,_ (5)


where _λ_ _S_ denotes the coefficient of the electrically small scenario, _d_ _SR_ and _d_ _RU_ represent


the distance of source-RIS and RIS-user links, respectively. A detailed discussion is given


in [50, Secs. IV-A, IV-B, IV-C].


_•_ **Electrically Large RISs:** In this asymptotic regime, the RIS is assumed to be large (ideally


infinitely large) in size compared with the transmission distances and the wavelength. In


this regime, the RIS can be approximated as a large flat mirror. Let us denote by _x_ 0 the


point of the RIS (if it exists) at which the first-order derivative of the total phase response


of the combined incident signal, reflected signal, and the surface reflection coefficient of


the RIS is equal to zero. In general, the path-loss asymptotically scales with the reciprocal


of a weighted sum of the distance between the transmitter and _x_ 0 and the distance between


_x_ 0 and the receiver. In addition, the received power is not dependent on the size of the RIS,


which is viewed as asymptotically infinite. This result substantiates the fact that the power


scaling law of the RIS is physically correct, since it does not grow to infinity as the size


of the RIS goes to infinity. This is because the scaling law and the behavior of the RIS are


different with respect to the electrically small regime. In this case, the path loss through


the RIS follows the “sum of distances” models, which can be approximated as:


_L_ ( _d_ _SR_ _, d_ _RU_ ) _≈_ _λ_ _L_ ( _d_ _SR_ + _d_ _RU_ ) _[−]_ [1] _,_ (6)


where _λ_ _L_ denotes the coefficient of the electrically large scenario. A detailed discussion is


24


given in [50, Secs. IV-A, IV-B, IV-C].


**2) Spatial Models**


Stochastic geometry (SG) tools are capable of capturing the location randomness of users thus


enabling the derivation of computable or closed-form expressions of key performance metrics.


Specifically, several spatial processes exist for modeling the locations of users in different wireless


networks, i.e., the homogeneous Poisson point process (HPPP) [65], [66], the Poisson cluster


process (PCP) [67], [68], the Binomial point process (BPP), as well as the Hard core point


process (HCPP) [69]. We list some promising approaches for analyzing the performance of


RIS-enhanced networks by using SG in Table IV. In [70], the RIS elements are employed on


obstacles, and it is assumed that randomly distributed users are located in the serving area of


the RISs. In [71], the objects are modeled by a modified random line process of fixed length


and with random orientations and locations. Therein, the probability that a randomly distributed


object that is coated with an RIS acts as a reflector for a given pair of transmitter and receiver


was investigated. In [72], PCP was invoked in the RIS-enhanced large-scale networks, where the


angle of reflection is constrained by the angle of incidence. Therefore, the randomly distributed


users and BSs are located at the same side of the RIS.


Table IV: Summary of RIS-enhanced SG Networks

|Approaches|Advantages|Disadvantages|Ref.|
|---|---|---|---|
|RIS-enhanced HPPP|Fairness-oriented design|Restricted user distribution|[5]|
|RIS-enhanced two layer HPPP|Coverage-hole enhancement|RISs are deployed at obstacles|[70]|
|HPPP conditioned on angle|Practical design|Not tractable|[73]|
|RIS-enhanced HPPP conditioned on angle|Practical design for RIS-enhanced networks|Complicated|–|



**3) Small-Scale Fading Models**


Currently, two main approaches have been used for analyzing the performance of RIS-aided


systems in small-scale fading channels: (1) the central-limit-theorem-based (CLT-based) distri

bution and (2) the use of approximated distributions.


_•_ **CLT-based Distribution:** Let us consider a single-antenna BS that communicates with a


single-antenna user with the aid of an RIS of _N_ elements. If the two received signals, from


the BS and from the RIS, can be coherently combined, the effective channel power gain is


given by
_H_ 2
�� **r** **Φg** + _h_ �� _,_


s _._ t _. β_ 1 _, · · ·, β_ _N_ = 1


_φ_ 1 _, · · ·, φ_ _N_ _∈_ [0 _,_ 2 _π_ ) _,_



25


(7)



where _h ∈_ C [1] _[×]_ [1], **g** _∈_ C _[M]_ _[×]_ [1], **r** _∈_ C _[M]_ _[×]_ [1] denote the channels of the BS-user, BS-RIS, and


RIS-user links, respectively. **Φ** = diag � _β_ 1 _e_ _[jφ]_ [1] _, β_ 2 _e_ _[jφ]_ [2] _, . . ., β_ _N_ _e_ _[jφ]_ _[N]_ [�] denotes the reflection

coefficient matrix of the RIS, where _{β_ 1 _, β_ 2 _. . ., β_ _N_ _}_ and _{φ_ 1 _, φ_ 2 _. . ., φ_ _N_ _}_ represent the


amplitude coefficients and phase shifts of the RIS elements, respectively. In this setup, the


CLT-based technique stands as an approximation tool for analyzing the performance in the


low-medium-SNR regime. This is due to the fact that the distribution of the probability


density function (PDF) in the range 0 to 0+ is not precise by using the CLT [74]. In


Rayleigh fading channels, the distribution of an RIS-enhanced link follows a modified


Bessel function [74]. Since both transmitter and RISs are part of the infrastructure, and


the RISs are typically positioned to exploit the LoS path with respect to the locations of


the transmitters and the receivers for increasing the received signal power, Zhang _et al._ [56]


studied Rician fading channels, and the analysis showed that the signal power follows a non

central chi-squared distribution with two degrees of freedom. Ding and Poor [75] proposed


an RIS-enhanced network, where RISs are utilized for effectively aligning the directions of


the users’ channel gains. By utilizing the CLT-based technique, Cheng _et al._ [76] studied


the multi-RIS network, where the channel distributions were investigated with or without


BS-user links.


_•_ **Approximated Distribution:** The exact distribution of the received SNR of the signal


reflected from an RIS is non-trivial to be obtained, and hence the use of approximated


distributions is often necessary. Qian _et al._ [55] proposed a simple approximated distribution


of the received SNR, and proved that the received SNR can be approximated by two (or


one) Gamma random variables and the sum of two scaled non-central chi-square random


variables. A prioritized signal enhancement design was proposed by Hou _et al._ [77], where


both the outage performance and ergodic rate of the user with the best channel gain were


calculated. Lyu and Zhang [78] proposed a single-input and single-output (SISO) network


26


with multiple randomly deployed RISs, and showed that the exact distribution in terms of


received signal power can be approximated by a Gamma distribution. Makarfi _et al._ [26]


proposed an RIS-enhanced network, whose equivalent channel is modeled by the Fisher

Snedecor _F_ distribution.


Based on the above mentioned contributions [26], [56], [74], [75], [77], [78], where only


approximated channel distributions are obtained, the exact channel distribution of RIS-enhanced


networks is still an open problem. Based on recent research results, for example, a fundamental


limitation lies in the calculation of the diversity order of RIS-enhanced networks under ideal


operating conditions and in the presence of hardware limitations, e.g., quantized phase shifts.


For example, the diversity order obtained by using the CLT-based distribution [74] is [1] 2 [in the]


high-SNR regime, whereas the diversity order is _[N]_ 3 [if an approximation based on the Gamma]


distribution is used, where _N_ denotes the number of RIS elements. However, the CLT-based and


Gamma-based distributions are not exact, making the performance analysis of RIS-enhanced


networks an interesting problem for future research. Furthermore, since the exact distribution


contains higher-order components, which approach zero in the high-SNR regime, most of the


previous contributions [4], [79] adopt the approximated distribution method for modeling small

scale fading channels, and the exact distribution of RIS-enhanced networks is still an open


problem. For example, recent exact results on the impact of phase noise on the diversity order


of RIS-enhanced transmission can be found in [80].


_B. Performance Analysis_


In this subsection, we briefly discuss currently available papers on the performance analysis


of RISs that are realized as large arrays of tiny and inexpensive antennas whose phase response


is locally optimized. By offering extra diversity in the spatial domain, multi-antenna techniques


are of significant importance. The application of multi-antenna enhanced RIS networks has


attracted substantial interest from academia [5], [77], [78] and industry [12], [81], [82]. Given


the increasing number of research contributions on RISs, its advantages are becoming more


clear, especially in terms of spectral efficiency (SE) and energy efficiency (EE) enhancement.


There are several key challenges for performance analysis in RIS-enhanced networks. One of


the main challenges is to evaluate the exact distributions of the cascade channels between the


27


BS and users through RISs. Another challenge is evaluating the effective channel gain after


passive beamforming at the RIS. Table V summarizes the existing contributions on RISs with


multiple antennas and illustrates their comparison. RIS-enhanced single user networks have been


analyzed in **Section III.A** . Hence, we turn out attention to RIS-enhanced multi-user networks.


A prioritized signal-enhancement-based (SEB) was proposed by Hou _et al._ [77], where passive


beamforming is designed for the user with the best channel gain, and all the other users rely on


RIS-enhanced beamforming.


Table V: Important contributions on RIS-enhanced networks. “DL” and “UL” represent downlink
and uplink, respectively. The “sum-rate gain” implies that the gain brought by invoking RIS
technique

|Ref.|Scenarios|Direction|Users|Main Objectives|Techniques|
|---|---|---|---|---|---|
|[5]|MIMO|DL|Multiple users|OP and throughput|Fairness SEB|
|[12]|SISO|DL|Single user|sum-rate gain|Compare with relay|
|[26]|SISO|UL|Single user|OP and throughput|Effective channel gain|
|[74]|SISO|DL|Single user|Effective channel gain|Compare with random phase shifting|
|[56]|SISO|DL|Single user|Effective channel gain|SEB|
|[75]|SISO|DL|Single user|OP|SEB|
|[77]|SISO|DL|Multiple users|OP and throughput|Prioritized SEB|
|[78]|SISO|DL|Single user|Effective channel gain|SEB|
|[55]|MIMO|DL|Single users|Effective channel gain|Random matrix theory and CLT|
|[83]|MIMO|DL|Multiple users|sum-rate gain|SEB|
|[82]|MIMO|DL|Multiple users|Interference cancellation|SCB and less constraint at RAs|
|[84]|SISO|UL|Multiple users|Sum-rate|Minimum required ﬁnite resolution|
|[85]|MISO|DL|Multiple users|Sum-rate|Multi-RIS distribution|
|[86]|MISO|DL|Multiple users|Sum-rate|Discrete phase shifts|



_•_ **RIS-enhanced Signal Enhancement Designs:** By assuming that multiple waves are co

phased at the users, the received signal can be significantly enhanced, which leads to the


following optimization problem:


_H_ 2
max �� **r** **Φg** + _h_ ��



s _._ t _. β_ 1 _, · · ·, β_ _N_ = 1


_φ_ 1 _, · · ·, φ_ _N_ _∈_ [0 _,_ 2 _π_ ) _._



(8)



In order to further enhance the SE of RIS-enhanced networks, multiple antenna techniques


can be employed at both the BS and users. Yuan _et al._ [85] proposed a cognitive-radio

based RIS-enhanced multiple-input and single-output (MISO) network, where both perfect


and imperfect channel state information (CSI) setups were considered. However, in many


research works, continuous amplitude coefficients and phase shifts are assumed at the RISs,


28


whilst in practice the phase shifts of RISs may not be continuous. Thus, You _et al._ [86]


proposed a discrete phase shifts model for a MISO enhanced RIS network. Zhang _et al._ [84]


then evaluated the required number of bits for finite-resolution RISs in an uplink SISO


network. Hou _et al._ [5] investigated an RIS-enhanced MIMO network, where a fairness


oriented design was considered by applying SG tools for modeling the impact of the users’


locations.


_•_ **RIS-enhanced Signal Cancellation Designs:** Another application of deploying RISs in


wireless networks is signal cancellation, where the reflected signals and the direct signals


can be destructively combined. The corresponding optimization problem can be formulated


as follows:
min �� **r** _H_ **Φg** + _h_ _I_ �� 2



s _._ t _. β_ 1 _, · · ·, β_ _N_ _≤_ 1


_φ_ 1 _, · · ·, φ_ _N_ _∈_ [0 _,_ 2 _π_ ) _._



(9)



where _h_ _I_ denotes the aggregate interference signals from other-cell BSs. By assuming that


both the inter-cell and intra-cell interferences are perfectly known, the optimal solution


to (9) is to adjust both the signal phase and amplitude coefficients of the BS-RIS-user


links to the opposite of the effective interference links with the same amplitude. By doing


so, some promising applications can be realized, e.g. RIS-enhanced PLS and interference


cancellation. On the one hand, by assuming that perfect CSI is available at the RIS controller,


the inter-cell and intra-cell interferences can be eliminated. On the other hand, considering


the PLS requirements, RISs also stand as a potential solution for cooperative jamming


techniques, i.e., RISs act as artificial noise sources. By adopting this approach, several


contributions have been made. Hou _et al._ [82] proposed an RIS-enhanced interference


cancellation technique in a MIMO network, where the inter-cluster interference can be


eliminated without active beamforming weights and detection vectors. Furthermore, this


work can be adopted for application to coordinated multi-point (CoMP) networks for inter

cell interference cancellation in cellular networks. Shi _et al._ [87] investigated an RIS

enhanced secure beamforming technique, where the secrecy rate of the legitimate user was


derived. Lyu _et al._ [88] investigated an RIS jamming scenario, where RISs act as jammers


29


for attacking legitimate communications without using any internal energy.


_C. Benchmark Schemes_


In order to assess the advantages and limitations of RISs, two benchmark transmission tech

nologies are usually considered: 1) surfaces with random phase shifts; and 2) relay networks.


_•_ **Random Phase Shifts:** RISs are capable of shifting the phase of the incident signal, and


hence multiple signals can be boosted or eliminated at the user side or at the BS side.


Hence, a well-accepted benchmark scheme to quantify the performance enhancement by


RIS elements is given by a surface that is not configurable and that can ideally be modeled


by a surface with random phase shifts [74].


_•_ **Relay Networks:** Generally speaking, relay-aided networks can be classified into two pairs


of classic relaying protocols, which are 1) FD and HD relay networks; and 2) AF and DF


relay networks. By assuming that the optimal power split strategies of both the AF and DF


relays are employed, the performance gain between RIS-enhanced and relay-aided networks


can be compared. Specifically, Bjornson _et al._ [12] compared the achievable data rate of


both RIS-enhanced and DF-relay-aided SISO network, where the BS-user links are blocked.


It was pointed out that when the number of tunable elements of the RISs is large enough,


an RIS-enhanced network is capable of outperforming a DF-relay-aided network. In an


effort to provide a comprehensive analysis for both RIS-enhanced and relay-aided networks,


Ntontin _et al._ [83] compared the system performance of classic maximal ratio transmission


(MRT) and maximal ratio combining (MRC) techniques. Fig. 11 illustrates the potential


benefits of RIS-enhanced networks compared with both HD-relay and FD-relay networks


in terms of network throughput [77]. Here, the performance of HD-relaying is obtained for


an equal time-split ratio. We can see that the network throughput gap between the RIS

enhanced network and the other pair of relay aided networks becomes smaller, when the


number of RIS elements increases. For example, when the number of RIS elements _N_ = 23


and the transmit power _P_ = 25 dBm, the proposed RIS-enhanced network is capable of


outperforming both FD and HD relay aided networks, which indicates that the RIS-enhanced


network becomes more competitive, when the number of RIS elements is large enough.


30



13


12


11


10


9


8


7


6


5


4









3

5 10 15 20 25 30 35 40 45 50

Number of RISs


Figure 11: Spectral efficiency of RIS-enhanced, FD-relay as well as HD-relay networks versus
the number of RIS elements. Please refer to [77] for simulation parameters.



Table VI provides a comparison between RIS-enhanced and relay aided networks in terms


of advantages and limitations.


Table VI: Comparison of the RIS-enhanced and Benchmarks. The “Power” Implies Additional
Power Supply at the RISs or at the Relay. The “CT” Denotes Concurrent Transmission

|Mode|Pros|Cons|Delay|Power|Interference|CT|
|---|---|---|---|---|---|---|
|RIS-enhanced|High EE, simple device|CSI must be perfectly known||||✓|
|HD relaying under AF protocol|No decoding at relay|Noise is ampliﬁed|✓|✓|||
|HD relaying under DF protocol|No self-interference|Latency is high|✓|✓|||
|FD relaying under AF protocol|No decoding at relay|Noise and interference are ampliﬁed||✓|Self-interference|✓|
|FD relaying under DF protocol|No latency|Rate ceiling occurs||✓|Self-interference|✓|
|MIMO relay|High SE|High cost, difﬁcult to realize at mmWave||✓|✓|✓|



_D. Discussions and Outlook_


Although previous research contributions have analyzed the approximated performance of


RIS-enhanced networks, there are still three major open research problems.


1) Path loss experiments for outdoor scenarios: Since only [64] reported experimental measure

ments of the path loss in free-space environments, the development of experimentally-validated


path-loss models in outdoor scenarios is an open research issue, especially in the presence of


reflecting and scattering objects.


2) Exact distributions: Current research methods for performance analysis are based on the


CLT-based distribution and approximated distribution [26], [76], which are, however, accurate


31


only in the high-SNR regime. More advanced and accurate analytical models are needed for


analyzing the diversity order and the network performance in the low-SNR regime.


3) Integrated application scenarios: To serve the desired users in different scenarios, signal


enhancement and signal cancellation designs were well investigated in recent research works [77],


[82], [83]. However, the desired signals and interference signals can be simultaneously enhanced


and mitigated, which constitutes an important future direction.


IV. RIS B EAMFORMING AND R ESOURCE A LLOCATION


As described in the previous section, deploying RISs enables high performance enhancements


in wireless networks. Motivated by this benefit, how to jointly design the transmit and passive


beamforming as well as the optimal allocation of the wireless resources has become an important


task for RIS-enhanced wireless networks. In this section, we first discuss the information-theoretic


performance limits of RISs. Then, we review recent research contributions with a particular


focus on the joint beamforming optimization and the resource allocation design. Along the


literature review, representative mathematical tools for facilitating the two types of design are


also discussed along with their benefits and drawbacks.


_A. Information-Theoretic Perspective_


In order to unveil the fundamental performance limits of RISs, several research works [61],


[62], [89] have been devoted to investigating the RIS performance gains from an information

theoretic perspective.


_•_ _Capacity-achieving design:_ In [61], Karasik _et al._ derived the capacity for an RIS-aided


single-input and multiple-output (SIMO) communication system. With finite input signal


constellations, it was proved that a joint information encoding scheme at both the transmitted


signals and the RIS configurations is necessary for achieving the channel capacity [61].


Based on this insight, the authors further proposed a practical transmission strategy by utiliz

ing layered encoding and successive cancellation decoding techniques. Numerical examples


showed that the proposed joint encoding scheme outperforms the conventional max-SNR


scheme.


32



7


6


5


4


3


2


1


0





7


6


5


4


3


2


1


0



















0 0.5 1 1.5 2 2.5 3

Rate at user 1 (bit/s/Hz)



0 0.5 1 1.5 2 2.5 3

Rate at user 1 (bit/s/Hz)



(a) Capacity regions with RIS-NOMA. (b) Rate regions with RIS-OMA.


Figure 12: Illustration of the capacity and rate regions for a random channel realization with
different RIS phase resolutions. _M_ _R_ denotes the number of RIS reflection elements. The full
parameter settings can be found in [62].


_•_ _Capacity region characterization:_ The capacity region of the fading SISO broadcast channel


(BC) was proved to be achieved by invoking the superposition coding (SC) at the transmitter


and the successive interference cancellation (SIC) at multiple receivers [90], i.e., employing


the NOMA transmission. Inspired by these results, Mu _et al._ [62] investigated the capacity


and rate regions of RIS-enhanced multi-user wireless communication systems achieved by


NOMA and orthogonal multiple access (OMA), respectively. The Pareto boundary of each


region was characterized by solving a series of sum rate maximization problems via the


rate-profile technique. As shown in Fig. 12(a) and Fig. 12(b), by deploying an RIS, the


NOMA capacity region and the OMA rate region can be improved. The capacity/rate regions


are further enlarged by employing more precise phase resolutions and larger numbers of


reflection elements. Furthermore, Zhang _et al._ [89] investigated the capacity region of the


multiple access channel (MAC) with two users in both centralized and distributed RIS


deployment strategies. The results demonstrated that the centralized RIS deployment strategy


can achieve a higher capacity gain than the distributed strategy.


33





Reflect link



RIS


User _n_





Figure 13: Illustration of joint transmit and passive beamforming design.


_B. Joint Transmit and Passive Beamforming Design_


_1) Optimization objectives:_ As shown in Fig. 13, an RIS is deployed to assist the transmission


between the BS and the users by passively reflecting the signals. The RIS reflection coefficients


can be adjusted by the BS through an RIS controller. Hence, the transmit beamforming at


the BS and the passive beamforming at the RIS have to be jointly designed to improve the


communication performance. In the following, we review the related research works in terms of


their considered optimization objectives.


_•_ _Transmit power minimization or EE maximization:_ In [58], Wu _et al._ minimized the transmit


power for an RIS-enhanced MISO system in both single-user and multi-user scenarios.


Alternating optimization (AO) based algorithms were developed to find locally-optimal


solutions. The passive beamforming was designed by invoking the semidefinite relaxation


(SDR) approach. It was revealed that an RIS can simultaneously enhance the desired signal


strength and mitigate the interference for the multi-user scenario. The same problem was


further investigated in [91] by taking discrete RIS phase shifts into consideration. The


optimal solutions were derived by applying the branch-and-bound method and exhaustive


search for single-user and multi-user scenarios, respectively. To reduce the computational


complexity, efficient successive refinement algorithms were further designed. It was shown


that the proposed low complexity algorithms are capable of achieving near-optimal perfor

mances. Han _et al._ [92] investigated physical-layer broadcasting in an RIS-aided network,


where the total transmit power for satisfying QoS requirements of all users was minimized.


34


Furthermore, Fu _et al._ [93] focused on an RIS-enhanced MISO downlink NOMA system,


where the transmit power was minimized by jointly optimizing the transmit and passive


beamforming vectors as well as user decoding orders. In order to overcome the drawbacks


of the SDR approach, an alternating difference-of-convex (DC) method was proposed for


handling the non-convex rank-one constraint. Zhu _et al._ [94] proposed an improved quasi

degradation condition for the RIS-enhanced MISO NOMA system to minimize the transmit


power. Under this condition, NOMA is shown to be able to outperform the zero-forcing


beamforming scheme. Zheng _et al._ [95] compared the minimum transmit power performance


between OMA and NOMA in a discrete phase shift RIS-enhanced SISO system. A near

optimal solution was obtained by applying the linear approximation initialization and the AO


method. The results showed that NOMA may perform worse than TDMA when the users


have symmetric deployments and rate requirements, which revealed the importance of user


pairing in the RIS-assisted NOMA system. Huang _et al._ [59] solved the EE maximization


problem in an RIS-enhanced multi-user MISO system, where a realistic RIS power con

sumption model was proposed in terms of the number of reflection elements and the phase


resolutions at the RIS. An AO-based algorithm was designed for addressing the formulated


problem, where the RIS phase shifts and the BS transmit power allocation were optimized by


invoking the gradient descent method and the fractional programming method. The results


demonstrated that the RIS achieves significantly better EE performance than the traditional


active relay-assisted communication. In contrast to the aforementioned works based on the


perfect CSI assumption, Zhou _et al._ [96] investigated the robust beamforming design for an


RIS-enhanced multi-user MISO system with imperfect CSI assumptions. The transmit power


was minimized while satisfying QoS requirements of all users under all possible channel


error realizations. The formulated non-convex problem was transformed into a sequence of


semidefinite programming (SDP) subproblems, where the CSI uncertainties and the non

convex unit-modulus constraints were handled by applying approximation transformations


and the convex-concave procedure [97], respectively. In [98], the robust beamforming design


was further studied under two channel error models, namely the bounded CSI error model


and the statistical CSI error model. The S-procedure and the Bernstein-Type inequality


were applied in each model. The results unveiled that the RIS may degrade the system


35


performance when the channel error is high. Zappone _et al._ [99] modeled the overhead for


carrying out the channel estimation and adjusting the RIS. Based on the proposed overhead


model, the EE of an RIS-empowered MIMO communication network was maximized by


jointly optimizing the RIS phase shifts as well as the transmitted and received filters.


_•_ _SE or capacity maximization:_ Yu _et al._ [100] considered the SE maximization problem in


an RIS-enhanced MISO system. Since the SDR approach only provides an approximate


solution [58], two efficient algorithms were proposed by invoking the fixed-point iteration


method and the manifold optimization method for the passive beamforming design. It was


demonstrated that the proposed algorithms can achieve a higher performance and consume a


lower complexity than the SDR approach. To solve the same problem, a branch-and-bound


algorithm was further proposed by Yu _et al._ [101], which is capable of obtaining a globally


optimal solution. Though suffering from an extremely high computational complexity, the


proposed branch-and-bound algorithm serves as a performance benchmark to verify the


effectiveness of existing suboptimal algorithms. In [102], Ning _et al._ focused on an RIS

enhanced downlink MIMO system to maximize the SE. The passive beamforming was


designed by using the sum of gains maximization principle and by utilizing the alternating


direction method of multipliers. In [103], Ying _et al._ considered an RIS-enhanced mmWave


hybrid MIMO system, where the phase shifts at the RIS were designed by leveraging the


angle information of the LoS BS-RIS channel. Moreover, Perovi´c _et al._ [9] investigated


RIS-assisted indoor mmWave communications, where two schemes were developed to


maximize the channel capacity. Zhang _et al._ [104] characterized the fundamental capacity


limit of RIS-aided point-to-point MIMO communication systems, by jointly optimizing


the RIS reflection coefficients and the MIMO transmit covariance matrix. The communi

cation capacity was maximized in both the narrowband transmission with frequency-flat


fading channels and the broadband orthogonal frequency division multiplexing (OFDM)


transmission over frequency-selective fading channels. Yang _et al._ [8] proposed a practical


transmission protocol by considering the channel estimation for an RIS-enhanced OFDM


system under frequency-selective channels. To reduce the required training overhead, the


RIS reflection elements were divided into multiple groups and only the combined channel


of each group has to be estimated. Based on the proposed grouping scheme, the achievable


36


rate was maximized by jointly optimizing the power allocation at the transmitter and the


phase shifts at the RIS with the AO method. In [105], You _et al._ designed a transmission


protocol by considering the channel estimation with discrete phase shifts at the RIS. To


reduce the channel estimation errors, a low complexity discrete fourier transform (DFT)

Hadamard based reflection pattern scheme was developed. The achievable data rate was


further maximized based on the estimated channel by designing the RIS phase shifts using


the proposed successive refinement algorithm.


_•_ _Sum rate maximization:_ In [106], Huang _et al._ maximized the sum rate in RIS-enhanced


multi-user MISO downlink communications. By employing the zero-forcing precoding at


the BS, the RIS reflection matrix and the power allocation were alternately optimized


with the aid of the majorization-minimization approach. Moreover, the weighted sum rate


maximization problem was investigated by Guo _et al._ [107]. Under the AO framework, the


transmit beamforming was obtained using the fractional programming method, and three


iterative algorithms were designed for optimizing the reflection coefficients in terms of


different types of RIS reflection elements. In [108], the asymptotic optimal discrete passive


beamforming solution was derived and a modulation scheme was proposed to maximize


the achievable sum rate for the RIS-enhanced multi-user MISO transmission. To further


enhance the performance, Jung _et al._ [108] designed a joint user scheduling and transmit


power control scheme, which can strike the tradeoff between the rate fairness and the


maximum sum rate among the users. Mu _et al._ [60] focused their attention on the sum


rate maximization problem in an RIS-enhanced MISO NOMA system with both ideal and


non-ideal assumptions of RIS elements. The non-convex rank-one constraint of the passive


beamforming design was handled by invoking the sequential rank-one constraint relaxation


approach, which is guaranteed to obtain a locally optimal rank-one solution. Instead of


optimizing the passive beamforming with the instantaneous CSI, Zhao _et al._ proposed a


two-timescale transmission protocol for maximizing the achievable average sum rate in


an RIS-enhanced multi-user system [109]. To reduce the channel training overhead and


complexity, the RIS phase shifts were firstly optimized with the statistical CSI. Then, the


transmit beamforming was designed with the instantaneous CSI and the optimized RIS


phase shifts.


37


_•_ _User fairness:_ Nadeem _et al._ [110] maximized the minimum SINR of an RIS-enhanced


MISO system, where the BS-RIS-user link was assumed to be a LoS channel. A determin

istic approximations was developed for the minimum SINR performance under the optimal


linear precoder by employing the random matrix theory. As a result, the RIS phase shifts


can be optimized using the channel’s large-scale statistics, which can significantly reduce


the overhead of the signal exchange [110]. Yang _et al._ [111] investigated the max-min


rate problem in an RIS-enhanced NOMA system in both single-antenna and multi-antenna


cases. A combined-channel-strength based user ordering scheme was proposed for achieving


a near-optimal performance.


Table VII: Contributions on joint transmit and passive beamforming design

|Ref.|Scenarios|Phase shifts|CSI|Main Objectives|Techniques/Characteristics|
|---|---|---|---|---|---|
|[58]|SU/MU-DL-MISO|Continuous|Perfect|Transmit power|AO-SDR-based algorithm and two stage algorithm|
|[91]|SU/MU-DL-MISO|Discrete|Perfect|Transmit power|Near-optimal ZF-based successive reﬁnement method|
|[92]|MU-DL-MISO|Continuous|Perfect|Transmit power|Physical-layer broadcasting|
|[93]|MU-DL-MISO NOMA|Continuous|Perfect|Transmit power|Alternating DC method|
|[94]|MU-DL-MISO NOMA|Continuous|Perfect|Transmit power|Improved quasi-degradation condition|
|[95]|MU-DL-SISO NOMA|Discrete|Perfect|Transmit power|Asymmetric and symmetric user pairing|
|[59]|MU-DL-MISO|Continuous|Perfect|EE|RIS power consumption model|
|[96]|MU-DL-MISO|Continuous|Imperfect|Transmit power|The worst-case robust beamforming design|
|[98]|MU-DL-MISO|Continuous|Imperfect|Transmit power|Imperfect cascaded channels at the transmitter|
|[99]|SU-DL-MIMO|Continuous|Estimated|EE|Overhead model for channel estimation and RIS conﬁguration|
|[100]|SU-DL-MISO|Continuous|Perfect|SE|Fixed point iteration and manifold optimization methods|
|[101]|SU-DL-MISO|Continuous|Perfect|SE|Branch-and-bound algorithm|
|[102]|SU-DL-MIMO|Continuous|Perfect|SE|Sum of gains principle|
|[103]|SU-DL-MIMO mmWave|Continuous|Perfect|SE|Broadband hybrid beamforming|
|[9]|SU-DL-MIMO mmWave|Continuous|Perfect|Channel capacity|RIS-assisted indoor mmWave environments|
|[104]|SU-DL-MIMO|Continuous|Perfect|Channel capacity|Frequency-ﬂat fading and frequency-ﬂat selective channels|
|[8]|SU-DL-SISO OFDMA|Continuous|Estimated|Achievable rate|RIS element grouping scheme|
|[105]|SU-UL-SISO|Discrete|Estimated|Achievable rate|Near-orthogonal DFT-Hadamard based reﬂection patterns|
|[106]|MU-DL-MISO|Continuous|Perfect|Sum rate|Majorization-minimization approach|
|[107]|MU-DL-MISO|Continuous/Discrete|Perfect|Weighted sum rate|Iterative algorithms with closed-form expressions|
|[108]|MU-DL-MISO|Discrete|Perfect|Sum rate|Interference-free modulation scheme|
|[60]|MU-DL-MISO NOMA|Continuous/Discrete|Perfect|Sum rate|Sequential rank-one constraint relaxation approach|
|[109]|MU-DL-MISO|Discrete|Statistical CSI|Sum rate|Low channel training overhead|
|[110]|MU-DL-MISO|Continuous|Statistical CSI|Max-min SINR|Signal exchange overhead reduction|
|[111]|MU-DL-SISO/MISO NOMA|Continuous|Perfect|Max-min rate|Near-optimal NOMA user ordering scheme|



All the aforementioned research contributions on the joint transmit and passive beamforming


design are summarized in Table VII. “SU” and “MU” represent single-user and multi-user,


respectively. “DL” and “UL” represent downlink and uplink, respectively.


38


_2) Approaches for passive beamforming design:_ An example of the joint transmit and passive


beamforming design problem can be formulated as follows


min _/_ max _f_ ( **w** _,_ _**θ**_ _|H_ ) (10a)

**w** _,_ _**θ**_


s _._ t _._ **w** _∈T,_ (10b)


_**θ**_ _∈P,_ (10c)


where _H_ denotes the set of given CSI, **w** and _**θ**_ denote the transmit beamforming vector and


passive beamforming vector, receptively, _T_ and _P_ denote the corresponding feasible set for **w**


and _**θ**_, respectively. Here, _f_ ( **w** _,_ _**θ**_ _|H_ ) denotes the objective function that depends on **w** and _**θ**_


given the CSI. Let _θ_ _i_ = _β_ _i_ _e_ _[jφ]_ _[i]_ be the _i_ th element of the passive beamforming vector _**θ**_ . Depending


on the specific implementation of the RIS, three case studies can be considered [60].


_•_ **Continuous amplitude and phase shift** : In this case, it is assumed that the amplitude and


phase shift of each RIS element can be adjusted continuously, which results in the following


feasible set.


_P_ 1 = _{β_ _i_ _, φ_ _i_ _|β_ _i_ _∈_ [0 _,_ 1] _, φ_ _i_ _∈_ [0 _,_ 2 _π_ ) _} ._ (11)


_•_ **Constant amplitude and continuous phase shift** : In this case, it is assumed that the


amplitude and phase shift of each RIS element are fixed, e.g., _β_ _i_ = 1, and can be adjusted


continuously, respectively. The corresponding feasible set is given by


_P_ 2 = _{β_ _i_ _, φ_ _i_ _|β_ _i_ = 1 _, φ_ _i_ _∈_ [0 _,_ 2 _π_ ) _} ._ (12)


_•_ **Constant amplitude and discrete phase shift** : In this case, it is assumed that the amplitude


and phase shift of each RIS element are fixed, e.g., _β_ _i_ = 1, and can be adjusted based on


a discrete set of values, respectively. The feasible set can be expressed as


_P_ 3 = _{β_ _i_ _, φ_ _i_ _|β_ _i_ = 1 _, φ_ _i_ _∈D},_ (13)



where _D_ = �0 _,_ [2] _N_ _[π]_



_N_ _[π]_ � and _N_ denotes the number of candidate phase shifts.




_[π]_ [2] _[π]_

_N_ _[,][ · · ·][,]_ [ (] _[N][ −]_ [1)] _N_



It is worth noting that the first two case studies are difficult to realize in practice. Due to the


39


hardware constraints in fact, it is quite challenging to realize continuous amplitude and phase


shift control. However, the first two case studies can be used to characterize the theoretical


performance upper bounds of RISs.


By inspection of the three case studies just considered, we evince that the joint beamforming


optimization problem is generally a non-convex problem since **w** and _**θ**_ are coupled together. The


existing algorithms for solving the non-convex joint beamforming optimization in RIS-assisted


wireless networks are mainly based on the AO method. The advantage of this approach is that,


given the passive beamforming vector, the transmit beamforming design becomes a conventional


problem, which has been extensively investigated. However, the passive beamforming design


under given transmit beamforming vectors is still a non-trivial task to tackle. The main challenges


to solve the problem include the unit modulus constraint and the discrete nature of the feasible set.


In the following, we list the approaches employed in current research contributions for optimizing


the passive beamforming. Table VIII summarizes the characteristics of those approaches.


Table VIII: Summary of approaches to passive beamforming design

|Approaches|Phase shifts|Advantages|Disadvantages|Ref.|
|---|---|---|---|---|
|SDR|Continuous|Relax to convex problem|Require rank-one solution construction|[58], [94], etc.|
|Quantization method|Discrete|Easy to implement|Substantial performance loss|[111]|
|Branch-and-bound|Continuous/Discrete|Optimal solution|Relatively high complexity|[91], [101]|
|Iterative algorithms|Continuous/Discrete|Good complexity-performance tradeoff|Performance depends on initialization|[91], [93], etc.|



_•_ **SDR:** A common method for handling the non-convex unit-modulus constraint is to trans

form the passive beamforming vector into a rank-one and positive semi-definite matrix. By


applying the SDR approach, which ignores the non-convex rank-one constraint, the original


non-convex problem becomes a convex SDP problem that can be solved by using many


efficient convex optimization tools. If the obtained matrix solution is not rank-one, Gaussian


randomization methods are usually used. However, the constructed rank-one solution is, in


general, suboptimal and it may be even infeasible for the original passive beamforming


design problem, which not only causes some performance degradation but cannot guarantee


the convergence of the AO-based iterative algorithm either.


_•_ **Quantization method:** Under the assumption of finite resolution phase shifts, one method



is to relax each discrete phase shift variable _φ_ _i_ _∈_ �0 _,_ [2] _N_ _[π]_




_[π]_ into a continuous

_N_ �




_[π]_ [2] _[π]_

_N_ _[,][ · · ·][,]_ [ (] _[N][ −]_ [1)] _N_


40


variable _φ_ _i_ _∈_ [0 _,_ 2 _π_ ). After solving the relaxed problem, the obtained continuous solutions


are quantized to their nearest discrete values. However, the quantization method may lead


to a substantial performance loss, especially for low-resolution phase shifts. Additionally, it


is worth mentioning that the non-convex unit-modulus constraints still exist after applying


the continuous relaxations.


_•_ **Branch-and-bound:** Due to the non-convex nature of the passive beamforming design


problem, it is challenging to obtain the optimal solution with standard convex optimization


techniques. The branch-and-bound approach has been applied for solving polynomial-time


(NP)-hard discrete and combinatorial optimization problems and some specific continuous


optimization problems. For example, the branch-and-bound approach was adopted for deriv

ing the optimal solution of the discrete passive beamforming design [91] and the continuous


passive beamforming design [101].


_•_ **Iterative algorithms:** The main idea of iterative algorithms is to obtain a locally optimal or


a high-quality suboptimal solution for the original problem at an acceptable computational


complexity. Some iterative algorithms were developed for the passive beamforming design,


such as the successive refinement algorithm [91], [104], [105], [107], the alternating DC


algorithm [93], the conjugate gradient search [59], the fixed point iteration method and


the manifold optimization method [100], and the sequential rank-one constraint relaxation


approach [60]. It was shown that these proposed iterative algorithms can achieve a good


tradeoff between performance and computational complexity.


_C. Resource Management in RIS-enhanced Networks_


_1) Resource allocation problems:_ In Fig. 14, a large-scale RIS-assisted transmission scenario


is considered, where multiple BSs serve multiple users with the aid of multiple RISs. In this


context, several key issues need to be discussed.


_•_ _Subchannel assignment:_ The bandwidth efficiency can be improved by properly allocating


users to different subchannels. If the RIS elements are not frequency-selective, one single


common RIS reflection matrix needs to be shared among all subchannels, which makes


the resulting optimization problems challenging to solve. To address this difficulty, Yang _et_


_al._ [112] proposed a dynamic passive beamforming scheme, where the resource blocks are


41



































BS 1 BS 2 BS 3 **BS index** RIS 1 RIS 2 **RIS index**



**BS index**



RIS 1 RIS 2



**(ii) user-BS association** **(iii) user-RIS association**

Figure 14: Illustration of resource management in large-scale RIS-assisted networks.


dynamically allocated to different user groups with varied RIS phase shifts over different


time slots. In [113], Zuo _et al._ investigated the joint subchannel assignment, power allo

cation, and passive beamforming design problem in a multi-channel downlink RIS-NOMA


network.


_•_ _User-RIS association:_ In multi-RIS assisted multi-user communications, how to associate


the users to different RISs is an interesting problem. The user-RIS association schemes


in general determine the overall network performance. Considering a multi-RIS assisted


massive MIMO system, Li _et al._ [114] found that the automatic interference cancelation


property holds for RISs with infinitely large sizes. Then, the considered max-min SINR


problem can be transformed into a user-RIS association problem, which was efficiently


solved by the proposed greedy search algorithm.


_•_ _Multi-cell RIS association:_ In multi-cell scenarios, the optimization problem becomes much


sophisticated for jointly considering the user-BS association, the user-RIS association, and


the subchannel assignment. In some initial works considering multi-cell scenarios [18],


[115], RISs were deployed to enhance the performance of cell-edge users for simultaneously


improving the received desired signal power and mitigating the received interference from


other cells.


42


_2) Approaches to resource allocation problems:_ Scheduling different users with different


subchannels/RISs/BSs is an NP-hard problem. Though the optimal solution can be obtained


by exhaustively searching over all possible association combinations, it requires a prohibitively


high computational complexity, especially for the large-scale networks in Fig. 14. Therefore,


low complexity and efficient algorithms have to be developed for striking a performance-versus

complexity tradeoff. In the following, we present some promising approaches, and discuss their


advantages and disadvantages. Table IX summarizes the characteristics of those approaches.


Table IX: Summary of approaches to resource allocation problems

|Approaches|Advantages|Disadvantages|Ref.|
|---|---|---|---|
|Binary relaxation|Relax to convex feasible set|Existence of performance gap|-|
|Matching theory|Achieve near-optimal performance|Require predeﬁned preference list|[113]|
|Heuristic algorithms|Flexible complexity-performance tradeoff|Unstable performance|[114]|



_•_ **Binary relaxation:** One idea is to relax the binary variable _α ∈{_ 0 _,_ 1 _}_ into the continuous


variable _α ∈_ [0 _,_ 1], where _α_ represents the user association state. By doing so, the non

convex integer constraint is relaxed to be convex, and conventional convex optimization


techniques can be applied for solving the relaxed problem. It is worth pointing out that the


relaxed problem might still be non-convex especially when the optimization variables are


highly-coupled. Additional efforts, such as utilizing the successive convex approximation


(SCA) method, are required to obtain an approximate solution. Moreover, this kind of re

laxation may result in a considerable performance loss between the original integer problem


and the relaxed one.


_•_ **Matching theory:** Matching theory is a powerful method developed for solving the combi

natorial user association problems. The user association combinational optimization prob

lem in RIS-enhanced networks can be modeled as a high dimensional Users-BSs-RISs

Subchannels matching problem. Though high dimensional matching is NP-hard, it can


be decomposed into several 2D matching subproblems, which can be efficiently solved.


For example, Zuo _et al._ [113] applied many-to-one matching theory for the subchannel


assignment in an RIS-enhanced NOMA system, which can achieve a near-optimal perfor

mance. However, leveraging matching theory requires to establish a predefined preference


list for both users and resources. As the channel conditions always fluctuate in RIS-enhanced


43


networks, these preference lists may need to be dynamically updated, which needs further


investigations.


_•_ **Heuristic algorithms:** For solving computationally complex problems, one commonly


employed method is to develop heuristic algorithms, where approximate solutions for the


original optimization problem can be obtained with an acceptable computational complexity.


In the aforementioned works, the greedy search based heuristic algorithm was designed for


solving the user-RIS association problem [114]. However, the performance of heuristic


algorithms is sensitive to the designed strategies, which is not always stable.


_D. Discussions and Outlook_


With the growing number of research contributions on RIS-enhanced communications, the


advantages of RISs have been verified in terms of SE, EE, and user fairness. However, most of


the existing treatises solved the non-convex joint beamforming optimization problem employing


the AO method, which decouples the joint transmit and passive beamforming design into two


subproblems. Though a high-quality suboptimal solution can be obtained, advanced optimization


techniques are required for solving this problem to characterize the optimal performance gain


introduced by RISs, which also provides an important benchmark for verifying the optimality of


any other low complexity suboptimal algorithms. Furthermore, in RIS-enhanced communication


systems, it is known that obtaining accurate CSI is rather challenging due to the nearly passive


working mode of RISs. Investigating the robust joint beamforming design and resource alloca

tion [96], [98] constitutes an important research direction for practical RIS implementations.


Besides passive beamforming, RISs also introduce the following DoFs, which can be further


exploited to reap the benefits of RISs in the future work.


_•_ _RIS deployment design:_ The reflection link via the RIS experiences a severer path loss


than the direct link. Therefore, the deployment location of the RIS has to be carefully


designed to achieve considerable performance enhancements. How to jointly optimize the


passive beamforming and the deployment location at the RIS as well as the wireless


resource allocation at the AP/BS is a non-trivial task, which deserves further research


efforts. In particular, one of prominent challenges is that the deployment location of the


RIS determines both the path loss and the LoS components of the reflection channels,


44


which causes the optimization variables to be highly-coupled. Therefore, efficient algorithms


need to be designed. An initial study [116] has investigated the optimal RIS deployment


strategy for both NOMA and OMA transmissions, which showed that asymmetric and


symmetric RIS deployment locations among users are preferable for NOMA and OMA,


respectively. Additionally, the RIS is usually deployed to avoid signal blockage to achieve


a LoS dominated channel, thus having a small path loss. However, such a LoS channel


based RIS deployment strategy may be ineffective, especially for RIS-assisted multi-user


communications. This is because the LoS dominated channels are low-rank, the resulting ill

condition channel matrices significantly limit the achievable capacity even with a relatively


small path loss. How to deploy the RIS to strike a balance between the path losses and the


Non-LoS (NLoS) components of channels is another interesting problem, which deserves


further research interests.


_•_ _Dynamical RIS configuration:_ In [61], [62], the authors revealed that, from an information

theoretic perspective, the capacity-achieving transmission schemes need to dynamically


adjusting the RIS. However, most of the existing research contributions assumed that the RIS


reflection coefficients can be only adjusted once for each channel coherence duration. Note


that one of the most typical application scenarios of RISs is to assist transmission of the


users, who are static or moving slowly in the vicinity of RISs. In this case, the duration of


one channel coherence block is usually tens of milliseconds, which is much larger than the


time duration for adjusting the RIS (e.g., 220 microseconds in [45]). Therefore, adjusting RIS


multiple times in one channel coherence duration, namely the dynamic RIS configuration,


is practically valid. This unique characteristic opens up new research opportunities, such


as dynamical passive beamforming and resource allocation schemes, which merit further


investigations.


V. M ACHINE L EARNING FOR RIS- ENHANCED C OMMUNICATION S YSTEMS


ML techniques have gained remarkable interests in wireless communications due to their


learning capability and large search-space [29], [117], [118]. We survey existing research contri

butions, which apply ML techniques for tackling challenges in RIS-enhanced wireless networks.


45


Finally, potential research challenges and opportunities of ML-empowered RIS systems are


presented.


_A. Motivations and Architecture for Integrating ML in RIS-enhanced Wireless Networks_


In this subsection, we first present the challenges of the conventional RIS-enhanced wireless


networks and the motivations for integrating ML in these networks, followed by the system


architecture of ML-empowered RIS-enhanced wireless networks.


To effectively exploit RISs for optimizing wireless networks, preliminary research contribu

tions have studied a number of technical challenges that include channel estimation/modeling,


joint transmit and passive beamforming design, as well as resource allocation from the BS


to the users. Powerful optimization techniques, such as convex optimization [107], iterative


algorithm [58], gradient descent approach [59], and alternating optimization algorithm [119] have


been adopted for addressing the aforementioned fundamental challenges. Although important


insights have been gained by these research contributions. the following limitations still exist in


conventional RIS-enhanced wireless networks:


_•_ The users are generally assumed to be static for simplicity, i.e., the dynamic mobility of users


is typically ignored. Another limitation in the existing literature is that the communication


environment is assumed to be perfectly known, the differentiation of users’ demand is always


ignored as well.


_•_ The RIS/BS are not capable of learning from the unknown environment or from the limited


feedback of the users. In practical applications of RISs in wireless networks, the system


parameters are treated as random variables, which naturally leads itself to the derivation of


insightful joint probability distributions conditioned on the users’ tele-traffic demand and


mobility. However, this is a highly dynamic stochastic environment, which is difficult for


employing conventional optimization approaches. Additionally, the feedback from the users


is usually resource-hungry and limited, which aggravates the challenges for the conventional


RIS-enhanced wireless networks.


_•_ Finally, instantaneous CSI of all the channels are assumed to be available at the BS.


However, CSI acquisition in RIS-enhanced wireless networks becomes more challenging


46


than that in the conventional relay systems due to the passive nature of RISs, which also


aggravates the challenge imposed on the conventional RIS-enhanced wireless networks.


_B. Deep Learning for RIS-enhanced Communication Systems_


Deep learning (DL) has shown great potentials to revolutionize communication systems. It


can be applied in diverse areas of RIS-enhanced wireless networks due to its powerful learning


capabilities [120]–[122].


The acquisition of timely and accurate CSI plays a pivotal role in wireless systems, especially


in MIMO networks. However, CSI acquisition becomes more challenging due to the large number


of antennas in massive MIMO systems [123]. In order to tackle this challenge, a number of


research contributions have adopted DL for estimating the CSI, especially for exploiting CSI


structures beyond linear correlations.


In contrast to the conventional AF relay-aided wireless networks, in RIS-enhanced wireless


networks, the RIS is a passive device, which is not capable of performing active transmis

sion/reception and signal processing [124]. In an effort to estimate a large number of unknown


parameters caused by RISs, Taha _et al._ [125] exploited the DL method for learning the RIS


reflection matrices directly from the sampled channel knowledge without any knowledge of


the RIS array geometry. Liu _et al._ [126] proposed a deep denoising neural network assisted


compressive channel estimation for RIS-assisted mmWave systems with a low training overhead.


Elbir _et al._ [127] presented a DL framework for channel estimation in the RIS-enhanced MIMO


system. It was shown that the proposed convolutional neural networks (CNNs)-based approach


achieves lower normalized mean-square-error (NMSE) and more robust performance than other


benchmarks.


The data-driven DL approach has the advantage of model-free representation or function


learning such that no explicit models of the complicated wireless channels are needed, at the


expense of requiring large amounts of training data and corresponding computational power.


Thus, the DL method can be adopted for estimating the CSI of RIS-enhanced wireless networks.


Apart from the aforementioned applications of DL in RIS-enhanced wireless networks, Huang


_et al._ [128] leveraged a deep neural network (DNN)-based approach in the indoor communication


environment for estimating the mapping between a user’s position and the configuration of the


47


RIS to maximize the received SNR. Additionally, DL can also be applied for learning the optimal


RIS phase shift configuration. Gao _et al._ [129] proposed a DL-based algorithm for optimally


designing the phase shift of the RIS by training the DL offline. It can be observed that the


proposed unsupervised learning mechanism outperforms the conventional optimization approach


in terms of computational complexity. Khan _et al._ [130] investigated the signal estimation


and detection in the RIS-enhanced wireless networks. A DL-based approach was proposed for


estimating channels and phase angles from a reflected signal received by an RIS. With the aid


of DL, the bit-error-rate (BER) performance of the system was improved.


_C. Reinforcement Learning for RIS-enhanced Communication Systems_


Reinforcement learning (RL) is a powerful AI paradigm that can be used to empower agents


by interacting with the environment. More explicitly, by exploiting the learning capability (e.g.,


learning from the environment, learning from the feedback of users, and learning from its


mistakes) of the RL model, the challenges encountered in the conventional RIS-enhanced wireless


networks may be mitigated, thus leading to improved performance.


The core idea of employing RL techniques in the RIS-enhanced wireless networks is that they


allow the BS/RIS to improve their service quality by learning from the environment, from their


historical experience, and from the feedback of the users [131]. More explicitly, RL models can be


used for supporting the BS/RIS (agents) in their interactions with the environment (states), whilst


finding the optimal behavior (actions) of the BS/RIS. Furthermore, the RL model can incorporate


farsighted system evolution (long-term benefits) instead of only focusing on current states. Thus,


it is applied for solving challenging problems in the RIS-enhanced wireless networks.


As illustrated in Fig. 15, the RL algorithms can be divided into three categories, namely,


value-based algorithms, policy-based algorithms, and actor-critic algorithms. Both advantages


and disadvantages exist in the RL algorithms. Since RISs have discrete phase shifts, the DQN


algorithm is more suitable for tackling the corresponding phase shift design problem.


To fully reap the benefits of deploying RISs in wireless networks, the joint transmit and passive


beamforming design of the RIS-enhanced system has been considered in MISO systems [142],


[143], OFDM-based systems [144], wireless security systems [145] and millimeter wave sys

tems [146] with the aid of RL algorithms. In contrast to the AO method, which alternately


48





























































|RL algorithm|Key features|Pros|Cons|
|---|---|---|---|
|Q-learning|Utilize the Q-table to<br>train Q-value offline|Can find the optimal policy<br>without requiring knowledge<br>about the environment|Only suitable for scenarios with<br>small state space and action<br>space|
|SARSA|Allows the agent<br>to approach the optimal<br>policy online|Allow the agent to choose<br>optimal actions at each time step<br>in a real-time fashion|Only suitable for scenarios with<br>small state space and action<br>space|
|DQN|Invoke NN to as an<br>approximator of Q-<br>function|Can be invoked in scenarios with<br>large state space and action space|Over estimation on action values|
|Double DQN|Invoke two Q-function to<br>select and evaluate Q-<br>values, respectively|Can solve the problem of over<br>estimation|Only discrete state space and<br>action space|
|Dueling DQN|Invoke two NNs to<br>estimate the action value<br>and state value|Have a faster convergence rate<br>than DQN model|High complexity|
|Noisy DQN|Adding Gaussian noise<br>layer in the DQN model|Enhance the exploring<br>performance|Only discrete state space and<br>action space|
|Distributional DQN|Invoke a distribution<br>function instead of<br>Expectation to update Q-<br>function|Higher accuracy than DQN in<br>terms of evaluating Q-function|Require the distribution<br>information of reward function|
|Asynchronous DQN|Multi-agent can train<br>parallelly|Have a faster learning speed than<br>DQN|High complexity|
|Retrace|Return-based off-policy<br>algorithm|More efficient than Q-learning|Only suitable for scenarios with<br>small state space and action<br>space|
|PG|Approximate a stochastic<br>policy directly using an<br>independent function<br>approximator with its<br>own parameters|Easy for application;|Sensitive to the parameter<br>setting|
|TRPO|Invoke a loss function to<br>find the optimized<br>parameters|Have a faster convergence rate<br>than value-based approaches|Can only be invoked in special<br>limiting cases; Have a low data<br>efficiency|
|PPO|Alternate between<br>sampling data through<br>interaction with the<br>environment|Easier and more general than<br>TRPO|Have a tradeoff between sample<br>complexity, simplicity and wall-<br>time|
|AC|Utilize critic to update the<br>value function parameters,<br>while utilize actor to<br>update the policy<br>parameters|Have the advantages of both<br>value-based algorithm and<br>policy-based algorithm|Have a lower convergence rate|
|A3C|Multi-agent can train<br>parallelly|Changes in the approximated<br>function get propagated much<br>more quickly|Have a higher variance|
|DDPG|Combination of AC and<br>DQN|Have the advantages of both<br>policy gradient and DQN|Higher complexity; Can not be<br>invoked in a random scenario|
|TD3|Invoke two critics to<br>update the value function<br>parameters|Overcome the problem of over<br>estimation|Higher complexity|
|SAC|Based on maximum<br>entropy|More stable; Enhance the<br>exploring performance|Higher complexity|


Figure 15: Key features, pros and cons of RL algorithms [131]–[141]. (PG represents Policy
Gradient, TRPO denotes Trust Region Policy Optimization, PPO represents Proximal Policy
Optimization, AC denotes Actor-Critic, A3C represents Asynchronous Advantage Actor-Critic,
DDPG denotes Deep Deterministic Policy Gradient, TD3 represents Twin Delayed DDPG, SAC
denotes Soft Actor-Critic)


optimizes the transmit beamforming at the BS and the passive beamforming at the RIS, the


RL-based solution is capable of simultaneously designing them. More explicitly, Huang _et_


49


_al._ [142] applied a deep deterministic policy gradient (DDPG) based algorithm for maximizing


the throughput by utilizing the sum rate as instant rewards for training the DDPG model. In the


proposed model, the continuous transmit beamforming and RIS phase shift were jointly optimized


with low complexity. Taha _et al._ [144] proposed a deep reinforcement learning (DRL) based


algorithm for maximizing the achievable communication rate by directly optimizing interaction


matrices from the sampled channel knowledge. In the proposed DRL model, only one beam was


utilized for each training episode. Thus, the training overhead was avoided, while the dataset


collection phase was not required. Zhang _et al._ [146] presented a DRL based algorithm for


maximizing the throughput with both perfect and imperfect CSI. A quantile regression method


was applied for modeling a return distribution for each state-action pair, which modeled the


intrinsic randomness in the MDP interaction between the RIS and communication environment.


Helin _et al._ [145] considered the application of RISs to PLS. The system secrecy rate was


maximized with the aid of the DRL model by jointly optimizing the beamforming and phase


shift matrices under different users’ QoS requirements and time-varying channel conditions.


Additionally, post-decision state and prioritized experience replay schemes were utilized to


enhance the learning efficiency and secrecy performance.


_D. A Novel Architecture of ML-empowered RIS-enhanced Wireless Networks_


As a benefit of the ML-based framework, many challenges in conventional wireless commu

nication networks have been circumvented, leading to enhanced network performance, improved


reliability, and agile adaptivity [117]. Fig. 16 illustrates a novel ML-empowered architecture for


RIS-enhanced wireless networks. As shown in this figure, RISs are installed on the facade


of a building for enhancing the wireless performance [3], [30]. The RIS is linked with a


controller, which controls the reflecting elements for hosting the functionality of phase-shifting


and amplitude absorption. A two-step approach is applied in the proposed ML-empowered RIS

enhanced wireless networks.


_•_ As illustrated in the data collection, data processing, and feature extraction parts in Fig. 16.


The associated user information (e.g., device type, position, data rate demand, mobility,


caching demand, and computing ability) is collected, stored, and processed. Thus, the users’


behaviors and requirements can be predicted for efficiently deploying and operating the RIS.


50







Decision making






|Phase shift policy|Col2|
|---|---|
|Deployment policy<br>|Deployment policy<br>|
|Resource allocation policy|Resource allocation policy|



User mobility Data integratio **n** **user 3**
information from OSN and clearning **base**























|Data collection<br>User mobility<br>information from OSN<br>User data demand<br>information from<br>telecom operator<br>Channel state<br>information from 3D<br>radio map|Col2|
|---|---|
|User mobility<br>information from OSN<br>User data demand<br>information from<br>telecom operator<br>Channel state<br>information from 3D<br>radio map<br>Data collection||
|User mobility<br>information from OSN<br>User data demand<br>information from<br>telecom operator<br>Channel state<br>information from 3D<br>radio map<br>Data collection||


|user 3<br>base<br>station domain<br>user 4<br>user 2 X Power<br>user 1 user 3 blocked<br>user 2<br>Frequency domain<br>Dynamic radio resource allocation user 1<br>Feature extraction<br>update<br>Online<br>Feature Data<br>refineme<br>extraction modeling<br>nt<br>Deep Learning|user 3<br>base<br>station domain<br>user 4<br>user 2 X Power<br>user 1 user 3 blocked<br>user 2<br>Frequency domain<br>Dynamic radio resource allocation user 1|
|---|---|
|**Deep Learning**<br>**Online**<br>**refineme**<br>**nt**<br>**Feature**<br>**extraction**<br>**Data**<br>**modeling**<br>update|**Deep Learning**<br>**Online**<br>**refineme**<br>**nt**<br>**Feature**<br>**extraction**<br>**Data**<br>**modeling**<br>update|


Figure 16: Architecture of ML-empowered RIS-enhanced wireless networks.


Meanwhile, the predicted information can be modified online with the currently collected


data as the input.


_•_ Given the extracted features, adaptive schemes are leveraged for controlling the RISs,


designing the phase shifts, resource allocation, and interference cancellation.


In the proposed ML-empowered RIS-enhanced wireless networks, RISs are capable of rapidly


adapting to the dynamic environment by learning both from the environment and from the


feedback of the users.


Research on the RIS deployment is fundamental and essential. However, there is a paucity of


research on the problem of RIS position determination. Additionally, current research contribu

tions mainly consider the performance optimization for both single-user and multi-user scenarios


by optimizing the phase shift and/or precoding solutions of the RIS-enhanced communication


systems [6], [18], [110], [147], [148].


Considering the RIS deployments based on the users’ mobility information and particular data


demand implicitly assumes that the long-term movement information and tele-traffic requirement


of users are capable of being learned/predicted. With this proviso, the deployment and control


method of RISs may be designed periodically for maximizing the long-term benefits and hence


reducing the control overhead. By considering the long-term mobility and data demand of users,


51


RIS-enhanced wireless networks become highly dynamic systems. Meanwhile, in an effort


to maximize the service quality in an unknown environment, RISs are supposed to learn by


interacting with the environment and adapting the control/deployment policy based on the limited


feedback of the users to overcome the uncertainty of the environment.


In this subsection, an RL-based model is presented to jointly design the deployment policy


and phase shift policy of RISs while considering the time-varying data demand of users. As


illustrated in Fig. 17, in the RL-based model, the BS acts as an agent. Since a controller is


installed, the BS can control both resource allocation policy for users and the RIS’s position


and phase shifts. At each timeslot, the BS periodically observes the state of the RIS-enhanced


system. The state space consists of the RIS phase shifts, the allocated power to each user, as


well as coordinates of both the RIS and users. An action is carried out by the BS for selecting


the optimal control policy. The actions contain changing positions and varying phase shifts of


the RIS, as well as varying the allocated power. The key underlying principle of the decision


policy is carrying out an action that makes the DQN model obtain the maximum Q-value at


each time slot. Following each action, the BS receives a penalty/reward, _r_ _t_, determined by the


formulated objective function.


_•_ **State of the RL model.** The state space consists of four parts: 1) the current phase shift


of each reflecting element at the RIS; 2) the current 3D position of the RIS; 3) the current


2D position of each user; 4) the current power allocated from the BS to each user.


_•_ **Action of the RL model.** The action space consists of three parts: 1) the variable quantity


of the _n_ -th reflecting element’s phase shift; 2) the moving direction and distance of the RIS;


3) the variable quantity of the _k_ -th user’s transmit power.


_•_ **Reward of the RL model.** The reward function is decided by the EE of the system. When


action taken by the BS improves EE, the BS obtains a reward. Otherwise, when a reduction


occurs in EE, the BS receives a penalty.


Fig. 18 characterizes the EE of the system in networks both with and without the assistance


of an RIS. The EE is defined as the ratio between the system achievable sum mean opinion


score (MOS) and the sum energy dissipation in Joule. It was shown that the EE of the system is


enhanced by employing an RIS. The RIS-barycenter line indicates that the RIS is placed at the


barycenter of all users. The RIS-random line indicates that the RIS is randomly deployed, while


52



Environment







Agent























|Col1|Action<br>Section|
|---|---|
|||


Figure 17: DRL model for RIS networks.



65


60


55


50


45


40


35


30


25


20


15


10




|Col1|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|Col10|Col11|
|---|---|---|---|---|---|---|---|---|---|---|
|||RIS|optimal||||||||
|||RIS-<br>RIS-<br>|barycen<br>random<br>|ter|||||||
||||||||||||
|||No-|RIS||||||||
||||||||||||
||||||||||||
||||||||||||
||||||||||||
||||||||||||
||||||||||||
||||||||||||



2 4 6 8 10 12 14 16 18 20

Transmit power (dBm)


Figure 18: EE with and without RIS [63].


the RIS-optimal line indicates that the RIS is deployed at the optimal position derived from the


proposed decaying double deep Q-network (D [3] QN) algorithm. The results of Fig. 18 confirm


that there exists an optimal position for the RIS as far as the EE of the RIS-enhanced system is


concerned. The performance of the RIS-enhanced system is improved by deploying the RIS at


the optimal position compared to the random deployment strategy and the strategy of placing it


at the barycenter.


53


_E. Other ML Techniques for RIS-enhanced Communication Systems_


Besides DL and RL techniques, a range of supervised learning and unsupervised learning


algorithms have been applied in the current generation wireless networks. Thus, these approaches


can also be adopted for tackling challenges in the RIS-enhanced wireless systems.


_1) Supervised Learning techniques for RIS-enhanced Communication Systems:_ As one of the


key branches of ML, powerful supervised learning techniques, such as regression, decision tree


and random forest, K-nearest neighbors (KNN), support vector machines (SVM), and Bayes


classification, have been adopted in diverse scenarios for tackling challenges such as spectrum


sensing [149], traffic/QoE prediction [150], channel/antenna selection [151], and networking


association [152]. In the RIS-enhanced wireless networks, supervised learning algorithms can also


be applied for solving the related problems with sufficient training data due to their advantages


of low complexity and fast convergence speed.


_2) Unsupervised Learning techniques for RIS-enhanced Communication Systems:_ In contrast


to the supervised learning techniques, unsupervised learning methods do not rely on prior


knowledge, which is not data-hungry. Thus, the unsupervised learning algorithms [153] such


as K-means clustering, expectation-maximization, principal component analysis (PCA), and


independent component analysis (ICA) can be applied in the RIS-enhanced wireless networks for


tackling challenges such as BS deployment, user clustering/association [154], channel/network


state detection [155], data aggregation [156], and interference cancellation [157].


_3) Federated Learning techniques for RIS-enhanced Communication Systems:_ Federated learn

ing, which explores training statistical models directly on remote devices, has become a focal


point in the area of large-scale ML and distributed optimization [158]. Since the federated


learning algorithm is trained at the edge in distributed networks, the inaccessibility of private


data is no longer a problem. Due to its privacy-preserving nature, federated learning algorithms


can be applied for the deployment and design of multiple RISs, where each RIS can act as a


distributed learner, trains its generated data and transfers its local model parameters instead of


the raw training dataset to an aggregating unit. Thus, the deployment and design policy can be


learned in a decentralized manner.


54


_F. Discussions and Outlook_


By exploiting ML learning capabilities, the aforementioned challenges encountered in RIS

enhanced wireless networks may be mitigated. This is due to the reason that RISs can learn by


interacting with the environment and adapt the control/deployment policy based on the feedback


of users to overcome the dynamic/uncertainty of the environment. It can be learned that the RL


model can incorporate farsighted system evolution instead of only focusing on current states,


which can reap long-term benefits for RIS-enhanced wireless networks. However, ML models


will also pose some new challenges, including the layer design for the DL model, the state

action construction, and the reward function design for the RL model. In addition, simultaneously


employing multiple RISs becomes more challenging due to the cooperation amongst RISs. Hence,


intelligent deployment and design for multi-RIS enhanced wireless networks is highly desired.


Finally, in the current ML models, either discrete or continuous state space is modeled for


optimizing parameters in RIS-enhanced wireless networks, while joint discrete and continuous


parameters exist in the networks. Hence, the joint discrete and continuous state space design in


ML-enabled RIS-enhanced wireless networks is still challenging and constitutes an interesting


topic.


VI. I NTEGRATING RIS S WITH O THER T ECHNOLOGIES T OWARDS 6G


Current research contributions have proved that RIS-enhanced wireless networks are capable


of obtaining tuned channel gains, improved QoS, enhanced coverage range, and reduced energy


dissipation. These significant performance enhancements can be applied to diverse wireless


communication networks. In this section, we identify the major issues and research opportunities


on the path to 6G associated with the integration of RISs and other emerging technologies, such


as NOMA, PLS, SWIPT, UAV-enabled wireless networks, and autonomous driving networks.


_A. NOMA and RIS_


In an effort to improve the SE and user connectivity of RIS-enhanced wireless networks,


power-domain NOMA technology is adopted, whose key idea is to superimpose the signals of


two users at different powers for exploiting the spectrum more efficiently by opportunistically


55


exploring the users’ different channel conditions [66], [159]. Li _et al._ [160] considered a MISO

NOMA downlink communication network for minimizing the total transmit power by jointly


designing the transmit precoding vectors and the reflecting coefficient vector. In [8], Yang _et al._


jointly optimized the phase shifts matrix of the RIS, as well as the power allocation from the


BS to the users. Thus, the minimum decoding SINR of all users was maximized for optimizing


the throughput of the system by considering user fairness. Ding _et al._ [75] proposed a novel


design of RIS assisted NOMA networks. It can be observed in [75] that, the directions of users’


channel vectors can be aligned with the aid of the RIS, which emphasizes the importance of


implementing NOMA technology. For an RIS-NOMA system, the core challenge is that the


decoding order is dynamically changed due to the configuration of phase shifts of the RIS.


Mu _et al._ [60] proposed an RIS-enhanced multiple-antenna NOMA transmission framework to


maximize the throughput of the system by considering the NOMA SIC decoding order condition.


The SCA technique and sequential rank-one constraint relaxation based algorithm were applied


to obtain a locally optimal solution. Ni _et al._ [161] proposed a resource allocation framework in


multi-cell RIS-NOMA networks, where the achievable sum rate was maximized by solving the


joint optimization problem of user association, sub-channel assignment, power allocation, phase


shifts design, and decoding order determination.


In contrast to the conventional MIMO-NOMA systems, RIS-NOMA technology can overcome


the challenges of the dynamic environment such as random fluctuation of wireless channels,


blocking, and user mobility in an energy-efficient manner. The NOMA system can obtain tuned


channel gains, improved fair resource allocation, enhanced coverage range, and high EE with


the aid of RISs [162]. However, NOMA also gives rise to new challenges when integrated with


RISs. For multi-antenna NOMA transmission, the decoding order is not determined by the users’


channel gains order, since additional decoding rate conditions need to be satisfied to guarantee


successful SIC [60]. Additionally, both the active beamforming and passive phase shift design


affect the decoding order among users and user clustering, which makes the decoding order


design, user clustering, and joint beamforming design highly-coupled in RIS-NOMA networks.


56


_B. PLS and RIS_


It has been shown that RISs are capable of simultaneously enhancing the desired signal


power at the intended user and mitigating the interference power at other unintended users [58].


Inspired by this result, several researchers explored the potential performance gain in the context


of PLS by applying the RIS [119], [163]–[169]. Yu _et al._ [163] considered an RIS-enhanced


multiple-input single-output single eavesdropper (MISOSE) channel, where the eavesdropper


is equipped with a single antenna. The secrecy rate was maximized by jointly optimizing the


transmit beamforming and the RIS phase shift matrix by using an AO-based algorithm. It was


demonstrated that the secrecy performance can be significantly improved by deploying the RIS.


Cui _et al._ [164] focused on the scenario where the eavesdropper has a better direct channel


condition than that of the legitimate receiver and they are also highly correlated in space, where


the achievable secrecy rate is rather limited in conventional communications. However, it was


shown that the direct signals and the reflected signals can be destructively combined at the


eavesdropper with the aid of RISs, thus significantly improving the secrecy rate. The same


problem was further investigated in [119], [166] by considering a multi-antenna eavesdropper or


legitimate receiver. Chu _et al._ [165] minimized the transmit power while satisfying the secrecy


rate requirement in the RIS-enhanced MISOSE system. Chen _et al._ [167] studied the minimum


secrecy rate maximization problem in the RIS-enhanced multi-user multiple-input single-output


multiple eavesdropper (MISOME) system by considering both the continuous and discrete RIS


phase shifts. Injecting artificial noise (AN) is an effective technique to enhance the secrecy


rate [170]. Motivated by this result, Guan _et al._ [168] examined the effectiveness of employing


AN in an RIS-enhanced MISOME system. The achievable secrecy rate was maximized by jointly


optimizing the transmit beamforming, the passive beamforming, and AN. The results verified


the necessity of using AN, especially for systems with a large number of eavesdroppers. Yu _et_


_al._ [169] considered an RIS-enhanced multi-user MISOME system under imperfect CSI with


the aim of maximizing the sum rate, subject to the maximum information leakage constraint.


An efficient AO-based algorithm was developed to optimize the transmit beamforming, the AN


covariance matrix, and the RIS phase shifts. Numerical results showed that significant secrecy


performance gains can be achieved by the RIS.


57


One critical issue of the RIS-enhanced PLS is that the joint design of transmit and passive


beamforming requires the CSI of both AP-eavesdropper and RIS-eavesdropper links, which is


quite challenging to obtain. This is because besides the nearly passive working mode of RISs, in


practice, eavesdroppers usually stay almost silent to hide their positions and only detect signals in


the air. Therefore, robust joint beamforming designs under the imperfect CSI of the eavesdropper


are essential to guarantee secure transmission. Moreover, given the uncertainty of eavesdroppers,


deploying the RIS may increase the probability of information leakage since the eavesdropper


can receive not only the direct signals from the AP but also the reflected signals from the RIS.


The situation may become even worse when there are multiple cooperative eavesdroppers. In this


case, setting a protected zone to establish an eavesdropper-exclusion area with carefully deployed


RISs would help to enhance the secrecy performance, which deserve further investigations.


_C. SWIPT and RIS_


SWIPT is an attractive technique for future IoT networks. However, the low EE at the


energy receivers is the main bottleneck in practical SWIPT systems. To overcome this limitation,


deploying the RIS is a promising solution and the RIS-assisted SWIPT has been investigated


in [19]–[22]. In [19], Wu _et al._ investigated an RIS-assisted SWIPT system, subject to individual


SINR requirements of information receivers. The weighted sum power received by energy


receivers was maximized by jointly optimizing the transmit and passive beamforming with the


proposed AO-based algorithm. Moreover, Tang _et al._ [20] maximized the minimum received


power among energy receivers. The results in [19] and [20] showed that deploying an RIS


can improve the energy harvesting efficiency. Pan _et al._ [21] studied the weighted sum rate


maximization problem in the RIS-assisted SWIPT MIMO system, subject to the energy harvesting


requirement of each energy receiver. A block coordinate descent (BCD)-based algorithm was


designed to find a Karush-Kuhn-Tucker (KKT) stationary point of the original optimization


problem. Wu _et al._ [22] extended the RIS-assisted SWIPT system into a multi-RIS case, where


the transmit power was minimized while satisfying the different QoS constraints at information


users and energy users. It was shown that the RIS enlarges the wireless power transfer range


and reduces the number of required energy beams.


58


Note that the above research contributions studied performance gain of deploying RISs for


SWIPT mainly from the communication perspective and ignored the EM characteristic of RISs.


As discussed in previous sections, there are substantial differences between the near-field region


and the far-field region of RISs. Therefore, sophisticated EM-based wireless power transfer


models are required for fully reaping the benefits of RISs, which need to be investigated in


future work.


_D. UAV and RIS_


RISs can be applied in UAV-enabled wireless networks, where UAVs are employed to com

plement and/or support the existing terrestrial cellular networks [171], [172]. An RIS enhances


the UAV coverage and service quality by compensating for the power loss over long distances,


as well as forming virtual LoS links between UAVs and mobile users via passively reflecting


their received signals. Li _et al._ [24] jointly optimized the UAV trajectory and the RIS phase


shifts in an iterative manner. It was shown in [24] that, the average achievable rates of the


users were significantly improved with the aid of RISs. On the other hand, RISs can also be


applied in UAV-aided wireless relay networks for enhancing performance. Zhang _et al._ [173]


considered the effective placement of a single UAV, which was equipped with an RIS to assist the


mmWave downlink transmission while considering user mobility. By jointly designing the UAV


trajectory and the RIS reflection parameters, a virtual LoS connection between the BS and users


was guaranteed. Thus, both the average data rate and the achievable downlink LoS probability


were improved. Yang _et al._ [174] derived the analytical expressions of outage probability, BER,


and average capacity by approximating the PDF of the instantaneous SNR in RIS-assisted UAV


relaying systems. Mu _et al._ [175] proposed a novel RIS-aided multi-UAV NOMA transmission


framework, where an RIS was deployed to enhance the desired signal strength between UAVs


with their served ground users while mitigating the inter-UAV interference. Liu _et al._ [176]


integrated UAVs in RIS-enhanced wireless networks for enhancing the service quality of the


UAV. With the aid of RISs, the energy consumption of the UAV was significantly reduced.


Due to the fact that UAVs are battery-powered, how to reduce their energy consumption is


one of the key challenges. The limited flight-time of UAVs (usually under 30 minutes) hampers


the wide commercial roll-out of UAV-aided networking. By deploying RISs, one can adjust the


59


RIS phase shift instead of controlling the UAV movement for forming virtual LoS links between


the UAV and the users. Therefore, the UAV can maintain hovering status only when the virtual


LoS links can not be formed even with the aid of the RIS. By invoking the aforementioned


protocol, the total energy consumption of the UAV is minimized, which in turn, maximizes the


UAV endurance. Additionally, by mounting a compact distributed laser charging (DLC) receiver


or wireless power transmission (WPT) receiver antenna inside the UAVs, while a DLC/WPT


transmitter is deployed on the ground or the building roof, the UAVs can be charged as long as


they are flying within the coverage range of the DLC/WPT transmitter [131], [177]. However,


the LoS connection between UAVs and the charging stations/vehicles have to be guaranteed,


which is challenging in the urban scenario when the LoS link between UAVs and charging


stations/vehicles are blocked by high-rise buildings with a high probability. RISs are capable


of smartly reconfiguring the wireless propagation environment by forming virtual LoS links


between UAVs and the charging stations/vehicles via passively reflecting their received signals.


Thus, the quality of charging service is enhanced with the aid of the RIS.


The RIS-enhanced UAV communication scenario is naturally a highly dynamic one, which


falls into the field of ML. When considering both the trajectory design of UAVs and the phase


shift design of RISs, the former one can be formed as a continuous state space while the latter


one is usually formed as a discrete one. Hence, how to simultaneously deal with both continuous


and discrete state space is challenging in ML-empowered RIS-enhanced UAV networks.


_E. Autonomous Driving/Connected Vehicles and RIS_


RISs can also be deployed in vehicle-to-infrastructure (V2I) assisted autonomous driving


systems, where V2I components are employed to complement the costly onboard units (OBUs).


V2I networks enable autonomous vehicles (AVs) or connected vehicles (CVs) to receive reliable


real-time traffic information from BSs, the information is collected by roadside base stations


(RBSs) and transmitted from RBSs to BSs, which facilitates the interaction among AVs/CVs and


road users, hence enhancing their safety and traffic efficiency [178]–[181]. Since AVs/CVs quality


and reliability are non-negotiable, the AVs/CVs system must be real-time, while the transmission


is supposed to be 100% reliable. However, the service quality of current V2I communication


systems cannot be guaranteed due to the complex channel terrain in the urban environment and


60


the complexity of road conditions, such as bad weather. Makarfi _et al._ [182] and Wang [183]


proved that the performance of vehicular networks can be significantly improved with the aid


of RISs. Since the RISs are made of EM material, which can be installed on key surfaces, such


as building facades, highway polls, advertising panels, vehicle windows, and even pedestrians’


clothes. With the massive deployment of RISs, a virtual LoS connection between the BSs and


AVs, as well as between the RSUs and BSs will be guaranteed, which enhances the reliability


of V2I communications.


In RIS-enhanced autonomous driving systems, the driving safety of AVs is the primary


consideration. In terms of safety, collisions have to be avoided, while the traffic rules also need


to obey. Additionally, in RIS-enhanced V2I-assisted autonomous driving systems, the wireless


service quality for AVs has to be guaranteed at each timeslot. Hence, how to improve the


reliability of RIS-enhanced autonomous driving systems is an open and challenging problem.


_F. Discussions and Outlook_


The studies of RISs have unveiled promising research opportunities, such as NOMA, PLS,


SWIPT, UAV-enabled wireless networks, and autonomous driving networks. Recent research


contributions have proved that RIS-enhanced wireless networks can achieve tuned channel gains,


improved QoS, enhanced coverage range, and reduced energy dissipation. However, the network


and beamforming designs are highly coupled due to dynamic control of the RIS phase shifts,


which brings challenges to these new research directions.


VII. C ONCLUSIONS, C HALLENGES, AND P OTENTIAL S OLUTIONS


_A. Concluding remarks_


In this paper, recent research works on RIS-enhanced wireless networks proposed for applica

tions to next-generation networks have been surveyed with an emphasis on the following aspects:


operating principles of RISs, performance evaluation of multi-antenna assisted RIS systems, joint


beamforming design and resource allocation for RISs, ML in RIS-enhanced wireless networks,


and their integration with other key 6G technologies. We have highlighted the advantages and


limitations of employing RISs for communication applications. Further research efforts are


needed to bridge the complex physical models of the different RISs implementations with widely


61


used communication models. We have considered the performance evaluation of multi-antenna


assisted RIS systems by systematically surveying existing designs for RIS-enhanced wireless


networks from the views of performance analysis, information theory, and optimization. In


addition, we have discussed existing research contributions that applying ML tools for tackling


the dynamic essence of the wireless environment such as random fluctuations of wireless channels


and user mobility. Design guidelines for ML-empowered RIS-enhanced wireless networks have


also been discussed. The benefits of integrating RISs with NOMA, UAV-terrestrial networks,


PLS, SWIPT, and AVs/CVs have been discussed. However, the research of RIS-enhanced wireless


networks is still at a very early stage and there are ample opportunities for important contributions


and advances in this field. Some of them are listed as follows.


_B. Challenges and Potential Solutions_


_1) CSI Acquisition:_ The acquisition of timely and accurate CSI plays a pivotal role in RIS

enhanced wireless systems, especially in MIMO-RIS and MISO-RIS networks. The majority


of current research contributions assume perfect CSI available at the BS, RISs controllers, as


well as the users. However, obtaining CSI in RIS-enhanced wireless networks is a non-trivial


task, which requires a non-negligible training overhead. Additionally, in RIS-assisted NOMA


networks, users in each cluster have to share the CSI with each other for implementing SIC.


However, due to the passive characteristic of RISs, the CSI acquisition and exchanging are


non-trivial. Potential solutions can be developed by employing DL methods for exploiting CSI


structures beyond linear correlations.


_2) Pareto-Optimization for Satisfying Multiple Objectives:_ In contrast to the conventional


wireless networks, RIS-enhanced wireless networks are characterized by more rapidly fluctuating


network topologies and more vulnerable communication links. Furthermore, RISs are more likely


to be deployed in an environment with heterogeneous mobility profiles. Hence, the networks


operate in a complex time-variant hybrid environment, where the classic mathematical models


have limited accuracy. Additionally, the challenging optimization problems encountered in RIS

enhanced wireless networks usually have to satisfy multiple objectives (e.g., delay, throughput,


BER, and power) in order to arrive at an attractive solution. To elaborate, by definition it is


only possible to improve any of the metrics considered at the cost of degrading at least one of


62


the others. The collection of Pareto-optimal points is referred to as the Pareto front. However,


determining the entire Pareto-front of optimal solutions is still challenging. Potential solutions


may be investigating near-real-time ML-aided Pareto-optimization for tackling the high-dynamic


adaptation of RIS-enhanced wireless networks.


R EFERENCES


[1] K. B. Letaief, W. Chen, Y. Shi, J. Zhang, and Y.-J. A. Zhang, “The roadmap to 6G: AI empowered wireless networks,”


_IEEE Commun. Mag._, vol. 57, no. 8, pp. 84–90, 2019.


[2] W. Saad, M. Bennis, and M. Chen, “A vision of 6G wireless systems: Applications, trends, technologies, and open


research problems,” _IEEE network_, vol. 34, no. 3, pp. 134–142, 2020.


[3] Q. Wu and R. Zhang, “Towards smart and reconfigurable environment: Intelligent reflecting surface aided wireless


network,” _IEEE Commun. Mag._, vol. 58, no. 1, pp. 106–112, 2020.


[4] Y. Cheng, K. H. Li, Y. Liu, K. C. Teh, and H. Vincent Poor, “Downlink and uplink intelligent reflecting surface aided


networks: NOMA and OMA,” _IEEE Trans. Wireless Commun._, doi: 10.1109/TWC.2021.3054841, 2021.


[5] T. Hou, Y. Liu, Z. Song, X. Sun, Y. Chen, and L. Hanzo, “MIMO assisted networks relying on intelligent reflective


surfaces,” _arXiv:1910.00959_, 2019.


[6] Y.-C. Liang, R. Long, Q. Zhang, J. Chen, H. V. Cheng, and H. Guo, “Large intelligent surface/antennas (LISA): Making


reflective radios smart,” _J. Commun. Inf. Netw._, vol. 4, no. 2, pp. 40–50, 2019.


[7] M. Di Renzo, A. Zappone, M. Debbah, M. S. Alouini, C. Yuen, J. de Rosny, and S. Tretyakov, “Smart radio environments


empowered by reconfigurable intelligent surfaces: How it works, state of research, and the road ahead,” _IEEE J. Sel. Areas_


_Commun._, vol. 38, no. 11, pp. 2450–2525, 2020.


[8] Y. Yang, B. Zheng, S. Zhang, and R. Zhang, “Intelligent reflecting surface meets OFDM: Protocol design and rate


maximization,” _IEEE Trans. Commun._, vol. 68, no. 7, pp. 4522–4535, 2020.


[9] N. S. Perovi´c, M. D. Renzo, and M. F. Flanagan, “Channel capacity optimization using reconfigurable intelligent surfaces


in indoor mmwave environments,” in _IEEE Proc. of International Commun. Conf. (ICC)_, 2020, pp. 1–7.


[10] C. Huang, S. Hu, G. C. Alexandropoulos, A. Zappone, C. Yuen, R. Zhang, M. D. Renzo, and M. Debbah, “Holographic


MIMO surfaces for 6G wireless networks: Opportunities, challenges, and trends,” _IEEE Wireless Commun._, vol. 27, no. 5,


pp. 118–125, 2020.


[11] M. Di Renzo, K. Ntontin, J. Song, F. H. Danufane, X. Qian, F. Lazarakis, J. de Rosny, D. . Phan-Huy, O. Simeone,


R. Zhang, M. Debbah, G. Lerosey, M. Fink, S. Tretyakov, and S. Shamai, “Reconfigurable intelligent surfaces vs. relaying:


Differences, similarities, and performance comparison,” _IEEE Open J. Commun. Soc._, vol. 1, pp. 798–807, 2020.


[12] E. Bj¨ornson, O. [¨] Ozdogan, and E. G. Larsson, “Intelligent reflecting surface versus decode-and-forward: How large surfaces [¨]


are needed to beat relaying?” _IEEE Wireless Commun. Lett._, vol. 9, no. 2, pp. 244–248, 2020.


[13] S. Zhou, W. Xu, K. Wang, M. Di Renzo, and M.-S. Alouini, “Spectral and energy efficiency of IRS-assisted MISO


communication with hardware impairments,” _IEEE Wireless Commun. Lett._, vol. 9, no. 9, pp. 1366–1369, 2020.


63


[14] Y. Cao and T. Lv, “Intelligent reflecting surface enhanced resilient design for MEC offloading over millimeter wave


links,” _arXiv:1912.06361_, 2019.


[15] T. Bai, C. Pan, Y. Deng, M. Elkashlan, A. Nallanathan, and L. Hanzo, “Latency minimization for intelligent reflecting


surface aided mobile edge computing,” _IEEE J. Sel. Areas Commun._, vol. 38, no. 11, pp. 2666–2682, 2020.


[16] Y. Cao and T. Lv, “Sum rate maximization for reconfigurable intelligent surface assisted device-to-device communications,”


_arXiv:2001.03344_, 2020.


[17] L. Yang, J. Yang, W. Xie, M. O. Hasna, T. Tsiftsis, and M. Di Renzo, “Secrecy performance analysis of RIS-aided


wireless communication systems,” _IEEE Trans. Veh. Technol._, vol. 69, no. 10, pp. 12 296–12 300, 2020.


[18] C. Pan, H. Ren, K. Wang, W. Xu, M. Elkashlan, A. Nallanathan, and L. Hanzo, “Multicell MIMO communications


relying on intelligent reflecting surfaces,” _IEEE Trans. Wireless Commun._, vol. 19, no. 8, pp. 5218–5233, 2020.


[19] Q. Wu and R. Zhang, “Weighted sum power maximization for intelligent reflecting surface aided SWIPT,” _IEEE Wireless_


_Commun. Lett._, vol. 9, no. 5, pp. 586–590, 2019.


[20] Y. Tang, G. Ma, H. Xie, J. Xu, and X. Han, “Joint transmit and reflective beamforming design for IRS-assisted multiuser


miso swipt systems,” in _IEEE Proc. of International Commun. Conf. (ICC)_, 2020, pp. 1–6.


[21] C. Pan, H. Ren, K. Wang, M. Elkashlan, A. Nallanathan, J. Wang, and L. Hanzo, “Intelligent reflecting surface enhanced


MIMO broadcasting for simultaneous wireless information and power transfer,” _IEEE J. Sel. Areas Commun._, vol. 38,


no. 8, pp. 1719–1734, 2020.


[22] Q. Wu and R. Zhang, “Joint active and passive beamforming optimization for intelligent reflecting surface assisted SWIPT


under QoS constraints,” _IEEE J. Sel. Areas Commun._, vol. 38, no. 8, pp. 1735–1748, 2020.


[23] H. Wang, Z. Zhang, B. Zhu, J. Dang, L. Wu, L. Wang, K. Zhang, and Y. Zhang, “Performance of wireless optical


communication with reconfigurable intelligent surfaces and random obstacles,” _arXiv:2001.05715_, 2020.


[24] S. Li, B. Duo, X. Yuan, Y.-C. Liang, M. Di Renzo _et al._, “Reconfigurable intelligent surface assisted UAV communication:


Joint trajectory design and passive beamforming,” _IEEE Wireless Commun. Lett._, vol. 9, no. 5, pp. 716–720, 2020.


[25] D. Ma, M. Ding, and M. Hassan, “Enhancing cellular communications for UAVs via intelligent reflective surface,” in


_Proc. IEEE Wireless Commun. Netw. Conf. (WCNC)_, 2020, pp. 1–6.


[26] A. U. Makarfi, K. M. Rabie, O. Kaiwartya, O. S. Badarneh, X. Li, and R. Kharel, “Reconfigurable intelligent surface


enabled iot networks in generalized fading channels,” in _IEEE Proc. of International Commun. Conf. (ICC)_, 2020, pp.


1–6.


[27] X. Mu, Y. Liu, L. Guo, J. Lin, and R. Schober, “Intelligent reflecting surface enhanced indoor robot path planning: A


radio map-based approach,” _IEEE Trans. Wireless Commun._, doi: 10.1109/TWC.2021.3062089, 2021.


[28] C. Liaskos, S. Nie, A. Tsioliaridou, A. Pitsillides, S. Ioannidis, and I. Akyildiz, “A new wireless communication paradigm


through software-controlled metasurfaces,” _IEEE Commun. Mag._, vol. 56, no. 9, pp. 162–169, 2018.


[29] H. Gacanin and M. Di Renzo, “Wireless 2.0: Toward an intelligent radio environment empowered by reconfigurable


meta-surfaces and artificial intelligence,” _IEEE Veh. Technol. Mag._, vol. 15, no. 4, pp. 74–82, 2020.


[30] M. Di Renzo, M. Debbah, D.-T. Phan-Huy, A. Zappone, M.-S. Alouini, C. Yuen, V. Sciancalepore, G. C. Alexandropoulos,


64


J. Hoydis, H. Gacanin _et al._, “Smart radio environments empowered by AI reconfigurable meta-surfaces: An idea whose


time has come,” _EURASIP J. Wireless Commun._, 2019.


[31] E. Basar, M. Di Renzo, J. De Rosny, M. Debbah, M.-S. Alouini, and R. Zhang, “Wireless communications through


reconfigurable intelligent surfaces,” _IEEE Access_, vol. 7, pp. 116 753–116 773, 2019.


[32] S. Gong, X. Lu, D. T. Hoang, D. Niyato, L. Shu, D. I. Kim, and Y. C. Liang, “Towards smart wireless communications


via intelligent reflecting surfaces: A contemporary survey,” _IEEE Commun. Surv. Tut._, vol. 22, no. 4, pp. 2283–2314,


2020.


[33] Q. Wu, S. Zhang, B. Zheng, C. You, and R. Zhang, “Intelligent reflecting surface aided wireless communications: A


tutorial,” _IEEE Trans. Commun._, doi: 10.1109/TCOMM.2021.3051897, 2021.


[34] D. R. Smith, O. Yurduseven, L. P. Mancera, P. Bowen, and N. B. Kundtz, “Analysis of a waveguide-fed metasurface


antenna,” _Physical Review Applied_, vol. 8, no. 5, p. 054048, 2017.


[35] B. O. Zhu, K. Chen, N. Jia, L. Sun, J. Zhao, T. Jiang, and Y. Feng, “Dynamic control of electromagnetic wave propagation


with the equivalent principle inspired tunable metasurface,” _Scientific reports_, vol. 4, no. 1, pp. 1–7, 2014.


[36] J. Y. Dai, J. Zhao, Q. Cheng, and T. J. Cui, “Independent control of harmonic amplitudes and phases via a time-domain


digital coding metasurface,” _Light: Science & Applications_, vol. 7, no. 1, p. 90, 2018.


[37] S. R. Rengarajan and Y. Rahmat-Samii, “The field equivalence principle: Illustration of the establishment of the non

intuitive null fields,” vol. 42, no. 4, pp. 122–128, 2000.


[38] V. S. Asadchy, M. Albooyeh, S. N. Tcvetkova, A. D´ıaz-Rubio, Y. Ra’di, and S. Tretyakov, “Perfect control of reflection


and refraction using spatially dispersive metasurfaces,” _Physical Review B_, vol. 94, no. 7, p. 075142, 2016.


[39] B. H. Fong, J. S. Colburn, J. J. Ottusch, J. L. Visher, and D. F. Sievenpiper, “Scalar and tensor holographic artificial


impedance surfaces,” vol. 58, no. 10, pp. 3212–3221, 2010.


[40] N. M. Estakhri and A. Al`u, “Wave-front transformation with gradient metasurfaces,” _Physical Review X_, vol. 6, no. 4, p.


041008, 2016.


[41] B. O. Zhu, J. Zhao, and Y. Feng, “Active impedance metasurface with full 360 reflection phase tuning,” _Scientific reports_,


vol. 3, p. 3059, 2013.


[42] N. K. Emani, A. V. Kildishev, V. M. Shalaev, and A. Boltasseva, “Graphene: a dynamic platform for electrical control


of plasmonic resonance,” _Nanophotonics_, vol. 4, no. 1, pp. 214–223, 2015.


[43] G. Gradoni and M. Di Renzo, “End-to-end mutual-coupling-aware communication model for reconfigurable intelligent


surfaces: An electromagnetic-compliant approach based on mutual impedances,” _IEEE Wireless Commun. Lett._, doi:


10.1109/LWC.2021.3050826, 2021.


[44] S. Abeywickrama, R. Zhang, Q. Wu, and C. Yuen, “Intelligent reflecting surface: Practical phase shift model and


beamforming optimization,” _IEEE Trans. Commun._, vol. 68, no. 9, pp. 5849–5863, 2020.


[45] V. Arun and H. Balakrishnan, “Rfocus: Beamforming using thousands of passive antennas,” in _17th USENIX Symposium_


_on Networked Systems Design and Implementation (NSDI 20)_, 2020, pp. 1047–1061.


[46] A. Welkie, L. Shangguan, J. Gummeson, W. Hu, and K. Jamieson, “Programmable radio environments for smart spaces,”


in _Proceedings of the 16th ACM Workshop on Hot Topics in Networks_, 2017, pp. 36–42.


65


[47] M. Dunna, C. Zhang, D. Sievenpiper, and D. Bharadia, “ScatterMIMO: enabling virtual MIMO with smart surfaces,” in


_Proceedings of the 26th Annual International Conference on Mobile Computing and Networking_, 2020, pp. 1–14.


[48] R. J. Bell, K. R. Armstrong, C. S. Nichols, and R. W. Bradley, “Generalized laws of refraction and reflection,” _JOSA_,


vol. 59, no. 2, pp. 187–189, 1969.


[49] J. Huang and J. A. Encinar, “Reflectarray antennas, a john wiley & sons,” _Inc., Publication_, 2008.


[50] M. Fadil H. Danufaneand Di Renzo, J. de Rosny, and S. Tretyakov, “On the path-loss of reconfigurable intelligent surfaces:


An approach based on green’s theorem applied to vector fields,” _arXiv:2007.13158_, 2020.


[51] R. C. Johnson, H. A. Ecker, and J. S. Hollis, “Determination of far-field antenna patterns from near-field measurements,”


_Proc. IEEE_, vol. 61, no. 12, pp. 1668–1694, 1973.


[52] H.-T. Chen, A. J. Taylor, and N. Yu, “A review of metasurfaces: physics and applications,” _Reports on progress in physics_,


vol. 79, no. 7, p. 076401, 2016.


[53] J. Huang and R. J. Pogorzelski, “A ka-band microstrip reflectarray with elements having variable rotation angles,” _IEEE_


_Trans. Antennas Propagat._, vol. 46, no. 5, pp. 650–656, 1998.


[54] J. Xu and Y. Liu, “A novel physics-based channel model for reconfigurable intelligent surface-assisted multi-user


communication systems,” _arXiv:2008.00619_, 2020.


[55] X. Qian, M. Di Renzo, J. Liu, A. Kammoun, and M. . S. Alouini, “Beamforming through reconfigurable intelligent


surfaces in single-user MIMO systems: SNR distribution and scaling laws in the presence of channel fading and phase


noise,” _IEEE Wireless Commun. Lett._, vol. 10, no. 1, pp. 77–81, 2021.


[56] Z. Zhang, Y. Cui, F. Yang, and L. Ding, “Analysis and optimization of outage probability in multi-intelligent reflecting


surface-assisted systems,” _arXiv:1909.02193_, 2019.


[57] A. A. A. Boulogeorgos and A. Alexiou, “Ergodic capacity analysis of reconfigurable intelligent surface assisted wireless


systems,” in _2020 IEEE 3rd 5G World Forum (5GWF)_, 2020, pp. 395–400.


[58] Q. Wu and R. Zhang, “Intelligent reflecting surface enhanced wireless network via joint active and passive beamforming,”


_IEEE Trans. Wireless Commun._, vol. 18, no. 11, pp. 5394–5409, 2019.


[59] C. Huang, A. Zappone, G. C. Alexandropoulos, M. Debbah, and C. Yuen, “Reconfigurable intelligent surfaces for energy


efficiency in wireless communication,” _IEEE Trans. Wireless Commun._, vol. 18, no. 8, pp. 4157–4170, 2019.


[60] X. Mu, Y. Liu, L. Guo, J. Lin, and N. Al-Dhahir, “Exploiting intelligent reflecting surfaces in NOMA networks: Joint


beamforming optimization,” _IEEE Trans. Wireless Commun._, vol. 19, no. 10, pp. 6884–6898, 2020.


[61] R. Karasik, O. Simeone, M. Di Renzo, and S. Shamai Shitz, “Beyond max-SNR: Joint encoding for reconfigurable


intelligent surfaces,” in _2020 IEEE International Symposium on Information Theory (ISIT)_, 2020, pp. 2965–2970.


[62] X. Mu, Y. Liu, L. Guo, J. Lin, and N. Al-Dhahir, “Capacity and optimal resource allocation for IRS-assisted multi-user


communication systems,” _IEEE Trans. Commun._, doi: 10.1109/TCOMM.2021.3062651, 2021.


[63] X. Liu, Y. Liu, Y. Chen, and H. V. Poor, “RIS enhanced massive non-orthogonal multiple access networks: Deployment


and passive beamforming design,” _IEEE J. Sel. Areas Commun._, vol. 39, no. 4, pp. 1057–1071, 2021.


[64] W. Tang, M. Z. Chen, X. Chen, J. Y. Dai, Y. Han, M. Di Renzo, Y. Zeng, S. Jin, Q. Cheng, and T. J. Cui, “Wireless


66


communications with reconfigurable intelligent surface: Path loss modeling and experimental measurement,” _IEEE Trans._


_Wireless Commun._, vol. 20, no. 1, pp. 421–439, 2021.


[65] Y. Liu, Z. Ding, M. Elkashlan, and H. V. Poor, “Cooperative non-orthogonal multiple access with simultaneous wireless


information and power transfer,” _IEEE J. Sel. Areas Commun._, vol. 34, no. 4, pp. 938–953, 2016.


[66] Y. Liu, Z. Qin, M. Elkashlan, Y. Gao, and L. Hanzo, “Enhancing the physical layer security of non-orthogonal multiple


access in large-scale networks,” _IEEE Trans. Wireless Commun._, vol. 16, no. 3, pp. 1656–1672, 2017.


[67] W. Yi, Y. Liu, and A. Nallanathan, “Modeling and analysis of D2D millimeter-wave networks with poisson cluster


processes,” _IEEE Trans. Commun._, vol. 65, no. 12, pp. 5574–5588, 2017.


[68] T. Hou, Y. Liu, Z. Song, X. Sun, and Y. Chen, “Exploiting NOMA for UAV communications in large-scale cellular


networks,” _IEEE Trans. Commun._, vol. 67, no. 10, pp. 6897–6911, 2019.


[69] H. ElSawy, E. Hossain, and M. Haenggi, “Stochastic geometry for modeling, analysis, and design of multi-tier and


cognitive cellular wireless networks: A survey,” _IEEE Commun. Surveys Tutorials_, vol. 15, no. 3, pp. 996–1019, 2013.


[70] M. A. Kishk and M. Alouini, “Exploiting randomly-located blockages for large-scale deployment of intelligent surfaces,”


_IEEE J. Sel. Areas Commun._, vol. 39, no. 4, pp. 1043–1056, 2021.


[71] M. Di Renzo and J. Song, “Reflection probability in wireless networks with metasurface-coated environmental objects:


An approach based on random spatial processes,” _EURASIP J. Wireless Commun._, no. 99, 2019.


[72] C. Zhang, W. Yi, and Y. Liu, “Reconfigurable intelligent surfaces aided multi-cell NOMA networks: A stochastic geometry


model,” _arXiv:2008.08457_, 2020.


[73] K. S. Ali, M. Haenggi, H. ElSawy, A. Chaaban, and M.-S. Alouini, “Downlink non-orthogonal multiple access (NOMA)


in poisson networks,” _IEEE Trans. Commun._, vol. 67, no. 2, pp. 1613–1628, 2018.


[74] Z. Ding, R. Schober, and H. V. Poor, “On the impact of phase shifting designs on IRS-NOMA,” _IEEE Wireless Commun._


_Lett._, vol. 9, no. 10, pp. 1596–1600, 2020.


[75] Z. Ding and H. Vincent Poor, “A simple design of IRS-NOMA transmission,” _IEEE Commun. Lett._, vol. 24, no. 5, pp.


1119–1123, 2020.


[76] Y. Cheng, K. H. Li, Y. Liu, K. C. Teh, and G. K. Karagiannidis, “Non-orthogonal multiple access (NOMA) with multiple


intelligent reflecting surfaces,” _arXiv:2011.00211_, 2020.


[77] T. Hou, Y. Liu, Z. Song, X. Sun, Y. Chen, and L. Hanzo, “Reconfigurable intelligent surface aided NOMA networks,”


_IEEE J. Sel. Areas Commun._, vol. 38, no. 11, pp. 2575–2588, 2020.


[78] J. Lyu and R. Zhang, “Spatial throughput characterization for intelligent reflecting surface aided multiuser system,” _IEEE_


_Wireless Commun. Lett._, vol. 9, no. 6, pp. 834–838, 2020.


[79] Z. Tang, T. Hou, Y. Liu, J. Zhang, and L. Hanzo, “Physical layer security of intelligent reflective surface aided NOMA


networks,” _arXiv:2011.03417_, 2020.


[80] P. Xu, G. Chen, Z. Yang, and M. D. Renzo, “Reconfigurable intelligent surfaces-assisted communications with discrete


phase shifts: How many quantization levels are required to achieve full diversity?” _IEEE Wireless Commun. Lett._, vol. 10,


no. 2, pp. 358–362, 2021.


[81] L. Dai, B. Wang, M. Wang, X. Yang, J. Tan, S. Bi, S. Xu, F. Yang, Z. Chen, M. Di Renzo _et al._, “Reconfigurable


67


intelligent surface-based wireless communications: Antenna design, prototyping, and experimental results,” _IEEE Access_,


vol. 8, pp. 45 913–45 923, 2020.


[82] T. Hou, Y. Liu, Z. Song, X. Sun, and Y. Chen, “MIMO-NOMA networks relying on reconfigurable intelligent surface:


A signal cancellation-based design,” _IEEE Trans. Commun._, vol. 68, no. 11, pp. 6932–6944, 2020.


[83] K. Ntontin, J. Song, and M. Di Renzo, “Multi-antenna relaying and reconfigurable intelligent surfaces: End-to-end SNR


and achievable rate,” _arXiv:1908.07967_, 2019.


[84] H. Zhang, B. Di, L. Song, and Z. Han, “Reconfigurable intelligent surfaces assisted communications with limited phase


shifts: How many phase shifts are enough?” _IEEE Trans. Veh. Technol._, vol. 69, no. 4, pp. 4498–4502, 2020.


[85] J. Yuan, Y. C. Liang, J. Joung, G. Feng, and E. G. Larsson, “Intelligent reflecting surface-assisted cognitive radio system,”


_IEEE Trans. Commun._, vol. 69, no. 1, pp. 675–687, 2021.


[86] C. You, B. Zheng, and R. Zhang, “Channel estimation and passive beamforming for intelligent reflecting surface: Discrete


phase shift and progressive refinement,” _IEEE J. Sel. Areas Commun._, vol. 38, no. 11, pp. 2604–2620, 2020.


[87] W. Shi, X. Zhou, L. Jia, Y. Wu, F. Shu, and J. Wang, “Enhanced secure wireless information and power transfer via


intelligent reflecting surface,” vol. 25, no. 4, pp. 1084–1088, 2021.


[88] B. Lyu, D. T. Hoang, S. Gong, D. Niyato, and D. I. Kim, “IRS-based wireless jamming attacks: When jammers can


attack without power,” _IEEE Wireless Commun. Lett._, vol. 9, no. 10, pp. 1663–1667, 2020.


[89] S. Zhang and R. Zhang, “Intelligent reflecting surface aided multiple access: Capacity region and deployment strategy,”


in _2020 IEEE 21st International Workshop on Signal Processing Advances in Wireless Communications (SPAWC)_, 2020,


pp. 1–5.


[90] L. Li and A. J. Goldsmith, “Capacity and optimal resource allocation for fading broadcast channels .I. ergodic capacity,”


_IEEE Trans. Inf. Theory_, vol. 47, no. 3, pp. 1083–1102, 2001.


[91] Q. Wu and R. Zhang, “Beamforming optimization for wireless network aided by intelligent reflecting surface with discrete


phase shifts,” _IEEE Trans. Commun._, vol. 68, no. 3, pp. 1838–1851, 2020.


[92] H. Han, J. Zhao, D. Niyato, M. D. Renzo, and Q. Pham, “Intelligent reflecting surface aided network: Power control for


physical-layer broadcasting,” in _IEEE Proc. of International Commun. Conf. (ICC)_, 2020, pp. 1–7.


[93] M. Fu, Y. Zhou, and Y. Shi, “Reconfigurable intelligent surface empowered downlink non-orthogonal multiple access,”


_IEEE Trans. Commun._, doi: 10.1109/TCOMM.2021.3066587, 2021.


[94] J. Zhu, Y. Huang, J. Wang, K. Navaie, and Z. Ding, “Power efficient IRS-assisted NOMA,” _IEEE Trans. Commun._,


vol. 69, no. 2, pp. 900–913, 2021.


[95] B. Zheng, Q. Wu, and R. Zhang, “Intelligent reflecting surface-assisted multiple access with user pairing: NOMA or


OMA?” _IEEE Commun. Lett._, vol. 24, no. 4, pp. 753–757, 2020.


[96] G. Zhou, C. Pan, H. Ren, K. Wang, M. Di Renzo, and A. Nallanathan, “Robust beamforming design for intelligent


reflecting surface aided miso communication systems,” _IEEE Wireless Commun. Lett._, vol. 9, no. 10, pp. 1658–1662,


2020.


[97] T. Lipp and S. Boyd, “Variations and extension of the convex–concave procedure,” _Optimization and Engineering_, vol. 17,


no. 2, pp. 263–287, 2016.


68


[98] G. Zhou, C. Pan, H. Ren, K. Wang, and A. Nallanathan, “A framework of robust transmission design for IRS-aided MISO


communications with imperfect cascaded channels,” _IEEE Trans. Signal Process._, vol. 68, pp. 5092–5106, 2020.


[99] A. Zappone, M. Di Renzo, F. Shams, X. Qian, and M. Debbah, “Overhead-aware design of reconfigurable intelligent


surfaces in smart radio environments,” _IEEE Trans. Wireless Commun._, vol. 20, no. 1, pp. 126–141, 2021.


[100] X. Yu, D. Xu, and R. Schober, “MISO wireless communication systems via intelligent reflecting surfaces,” in _Proc._


_IEEE/CIC Int. Conf. Commun. China (ICCC)_, 2019, pp. 735–740.


[101] ——, “Optimal beamforming for MISO communications via intelligent reflecting surfaces,” in _2020 IEEE 21st_


_International Workshop on Signal Processing Advances in Wireless Communications (SPAWC)_, 2020, pp. 1–5.


[102] B. Ning, Z. Chen, W. Chen, and J. Fang, “Beamforming optimization for intelligent reflecting surface assisted mimo: A


sum-path-gain maximization approach,” _IEEE Wireless Commun. Lett._, vol. 9, no. 7, pp. 1105–1109, 2020.


[103] K. Ying, Z. Gao, S. Lyu, Y. Wu, H. Wang, and M. Alouini, “GMD-based hybrid beamforming for large reconfigurable


intelligent surface assisted millimeter-wave massive MIMO,” _IEEE Access_, vol. 8, pp. 19 530–19 539, 2020.


[104] S. Zhang and R. Zhang, “Capacity characterization for intelligent reflecting surface aided MIMO communication,” _IEEE_


_J. Sel. Areas Commun._, vol. 38, no. 8, pp. 1823–1838, 2020.


[105] C. You, B. Zheng, and R. Zhang, “Intelligent reflecting surface with discrete phase shifts: Channel estimation and passive


beamforming,” in _IEEE Proc. of International Commun. Conf. (ICC)_, 2020, pp. 1–6.


[106] C. Huang, A. Zappone, M. Debbah, and C. Yuen, “Achievable rate maximization by passive intelligent mirrors,” in _Proc._


_IEEE Int. Conf. Acoust., Speech Signal Process. (ICASSP)_, 2018, pp. 3714–3718.


[107] H. Guo, Y.-C. Liang, J. Chen, and E. G. Larsson, “Weighted sum-rate optimization for intelligent reflecting surface


enhanced wireless networks,” _arXiv:1905.07920_, 2019.


[108] M. Jung, W. Saad, M. Debbah, and C. S. Hong, “On the optimality of reconfigurable intelligent surfaces (RISs): Passive


beamforming, modulation, and resource allocation,” _IEEE Trans. Wireless Commun._, doi: 10.1109/TWC.2021.3058366,


2021.


[109] M. M. Zhao, Q. Wu, M. J. Zhao, and R. Zhang, “Intelligent reflecting surface enhanced wireless networks: Two-timescale


beamforming optimization,” _IEEE Trans. Wireless Commun._, vol. 20, no. 1, pp. 2–17, 2021.


[110] A. Kammoun, A. Chaaban, M. Debbah, M.-S. Alouini _et al._, “Asymptotic max-min SINR analysis of reconfigurable


intelligent surface assisted MISO systems,” _IEEE Trans. Wireless Commun._, vol. 19, no. 12, pp. 7748–7764, 2020.


[111] G. Yang, X. Xu, and Y.-C. Liang, “Intelligent reflecting surface assisted non-orthogonal multiple access,”


_arXiv:1907.03133_, 2019.


[112] Y. Yang, S. Zhang, and R. Zhang, “IRS-enhanced OFDMA: Joint resource allocation and passive beamforming


optimization,” _IEEE Wireless Commun. Lett._, vol. 9, no. 6, pp. 760–764, 2020.


[113] J. Zuo, Y. Liu, Z. Qin, and N. Al-Dhahir, “Resource allocation in intelligent reflecting surface assisted NOMA systems,”


_IEEE Trans. Commun._, vol. 68, no. 11, pp. 7170–7183, 2020.


[114] X. Li, J. Fang, F. Gao, and H. Li, “Joint active and passive beamforming for intelligent reflecting surface-assisted massive


MIMO systems,” _arXiv:1912.00728_, 2019.


69


[115] H. Xie, J. Xu, and Y. F. Liu, “Max-min fairness in IRS-aided multi-cell MISO systems with joint transmit and reflective


beamforming,” _IEEE Trans. Wireless Commun._, vol. 20, no. 2, pp. 1379–1393, 2021.


[116] X. Mu, Y. Liu, L. Guo, J. Lin, and R. Schober, “Joint deployment and multiple access design for intelligent reflecting


surface assisted networks,” _IEEE Trans. Wireless Commun._, doi: 10.1109/TWC.2021.3075885, 2020.


[117] X. Liu, M. Chen, Y. Liu, Y. Chen, S. Cui, and L. Hanzo, “Artificial intelligence aided next-generation networks relying


on UAVs,” _IEEE Wireless Commun._, vol. 28, no. 1, pp. 120–127, 2021.


[118] C.-X. Wang, M. Di Renzo, S. Stanczak, S. Wang, and E. G. Larsson, “Artificial intelligence enabled wireless networking


for 5G and beyond: Recent advances and future challenges,” _IEEE Wireless Commun._, vol. 27, no. 1, pp. 16–23, 2020.


[119] H. Shen, W. Xu, S. Gong, Z. He, and C. Zhao, “Secrecy rate maximization for intelligent reflecting surface assisted


multi-antenna communications,” _IEEE Commun. Lett._, vol. 23, no. 9, pp. 1488–1492, 2019.


[120] A. Zappone, M. Di Renzo, M. Debbah, T. T. Lam, and X. Qian, “Model-aided wireless artificial intelligence: Embedding


expert knowledge in deep neural networks for wireless system optimization,” _IEEE Veh. Technol. Mag._, vol. 14, no. 3,


pp. 60–69, 2019.


[121] Z. Qin, H. Ye, G. Y. Li, and B. F. Juang, “Deep learning in physical layer communications,” _IEEE Wireless Commun._,


vol. 26, no. 2, pp. 93–99, 2019.


[122] A. Zappone, M. Di Renzo, and M. Debbah, “Wireless networks design in the era of deep learning: Model-based, AI-based,


or both?” _IEEE Trans. Commun._, vol. 67, no. 10, pp. 7331–7376, 2019.


[123] C.-K. Wen, W.-T. Shih, and S. Jin, “Deep learning for massive MIMO CSI feedback,” _IEEE Wireless Commun. Lett._,


vol. 7, no. 5, pp. 748–751, 2018.


[124] J. Chen, Y.-C. Liang, H. V. Cheng, and W. Yu, “Channel estimation for reconfigurable intelligent surface aided multi-user


MIMO systems,” _arXiv:1912.03619_, 2019.


[125] A. Taha, M. Alrabeiah, and A. Alkhateeb, “Enabling large intelligent surfaces with compressive sensing and deep learning,”


_IEEE Access_, vol. 9, pp. 44 304–44 321, 2021.


[126] S. Liu, Z. Gao, J. Zhang, M. Di Renzo, and M. Alouini, “Deep denoising neural network assisted compressive channel


estimation for mmWave intelligent reflecting surfaces,” _IEEE Trans. Veh. Technol._, vol. 69, no. 8, pp. 9223–9228, 2020.


[127] A. M. Elbir, A. Papazafeiropoulos, P. Kourtessis, and S. Chatzinotas, “Deep channel learning for large intelligent surfaces


aided mm-Wave massive MIMO systems,” _IEEE Wireless Commun. Lett._, vol. 9, no. 9, pp. 1447–1451, 2020.


[128] C. Huang, G. C. Alexandropoulos, C. Yuen, and M. Debbah, “Indoor signal focusing with deep learning designed


reconfigurable intelligent surfaces,” in _2019 IEEE 20th International Workshop on Signal Processing Advances in Wireless_


_Communications (SPAWC)_, 2019, pp. 1–5.


[129] J. Gao, C. Zhong, X. Chen, H. Lin, and Z. Zhang, “Unsupervised learning for passive beamforming,” _IEEE Commun._


_Lett._, vol. 24, no. 5, pp. 1052–1056, 2020.


[130] S. Khan and S. Y. Shin, “Deep-learning-aided detection for reconfigurable intelligent surfaces,” _arXiv:1910.09136_, 2019.


[131] X. Liu, Y. Liu, Y. Chen, and L. Hanzo, “Trajectory design and power control for multi-UAV assisted wireless networks:


A machine learning approach,” _IEEE Trans. Veh. Technol._, vol. 68, no. 8, pp. 7957–7969, 2019.


70


[132] H. Van Hasselt, A. Guez, and D. Silver, “Deep reinforcement learning with double Q-learning,” in _Thirtieth AAAI_


_Conference on Artificial Intelligence_, 2016.


[133] Z. Wang, T. Schaul, M. Hessel, H. Van Hasselt, M. Lanctot, and N. De Freitas, “Dueling network architectures for deep


reinforcement learning,” _arXiv:1511.06581_, 2015.


[134] M. Fortunato, M. G. Azar, B. Piot, J. Menick, I. Osband, A. Graves, V. Mnih, R. Munos, D. Hassabis, O. Pietquin _et al._,


“Noisy networks for exploration,” _arXiv:1706.10295_, 2017.


[135] W. Dabney, M. Rowland, M. G. Bellemare, and R. Munos, “Distributional reinforcement learning with quantile regression,”


in _Thirty-Second AAAI Conference on Artificial Intelligence_, 2018.


[136] V. Mnih, A. P. Badia, M. Mirza, A. Graves, T. Lillicrap, T. Harley, D. Silver, and K. Kavukcuoglu, “Asynchronous


methods for deep reinforcement learning,” in _International conference on machine learning_, 2016, pp. 1928–1937.


[137] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, “Proximal policy optimization algorithms,”


_arXiv:1707.06347_, 2017.


[138] P. H¨am¨al¨ainen, A. Babadi, X. Ma, and J. Lehtinen, “PPO-CMA: Proximal policy optimization with covariance matrix


adaptation,” _arXiv:1810.02541_, 2018.


[139] T. P. Lillicrap, J. J. Hunt, A. Pritzel, N. Heess, T. Erez, Y. Tassa, D. Silver, and D. Wierstra, “Continuous control with


deep reinforcement learning,” _arXiv:1509.02971_, 2015.


[140] T. Haarnoja, A. Zhou, P. Abbeel, and S. Levine, “Soft actor-critic: Off-policy maximum entropy deep reinforcement


learning with a stochastic actor,” _arXiv:1801.01290_, 2018.


[141] N. C. Luong, D. T. Hoang, S. Gong, D. Niyato, P. Wang, Y.-C. Liang, and D. I. Kim, “Applications of deep reinforcement


learning in communications and networking: A survey,” _IEEE Commun. Surv. Tut._, vol. 21, no. 4, pp. 3133–3174, 2019.


[142] C. Huang, R. Mo, and C. Yuen, “Reconfigurable intelligent surface assisted multiuser MISO systems exploiting deep


reinforcement learning,” _IEEE J. Sel. Areas Commun._, vol. 38, no. 8, pp. 1839–1850, 2020.


[143] K. Feng, Q. Wang, X. Li, and C.-K. Wen, “Deep reinforcement learning based intelligent reflecting surface optimization


for MISO communication systems,” _IEEE Wireless Commun. Lett._, vol. 9, no. 5, pp. 745–749, 2020.


[144] A. Taha, Y. Zhang, F. B. Mismar, and A. Alkhateeb, “Deep reinforcement learning for intelligent reflecting surfaces:


Towards standalone operation,” in _2020 IEEE 21st International Workshop on Signal Processing Advances in Wireless_


_Communications (SPAWC)_, 2020, pp. 1–5.


[145] H. Yang, Z. Xiong, J. Zhao, D. Niyato, L. Xiao, and Q. Wu, “Deep reinforcement learning based intelligent reflecting


surface for secure wireless communications,” _IEEE Trans. Wireless Commun._, doi: 10.1109/TWC.2020.3024860, 2020.


[146] Q. Zhang, W. Saad, and M. Bennis, “Millimeter wave communications with an intelligent reflector: Performance


optimization and distributional reinforcement learning,” _arXiv:2002.10572_, 2020.


[147] J. Ye, S. Guo, and M. Alouini, “Joint reflecting and precoding designs for SER minimization in reconfigurable intelligent


surfaces assisted MIMO systems,” _IEEE Trans. Wireless Commun._, vol. 19, no. 8, pp. 5561–5574, 2020.


[148] M. Jung, W. Saad, and G. Kong, “Performance analysis of large intelligent surfaces (LISs): Uplink spectral efficiency


and pilot training,” _arXiv:1904.00453_, 2019.


71


[149] K. Umebayashi, M. Kobayashi, and M. L´opez-Ben´ıtez, “Efficient time domain deterministic-stochastic model of spectrum


usage,” _IEEE Trans. Wireless Commun._, vol. 17, no. 3, pp. 1518–1527, 2017.


[150] Z. Feng, X. Li, Q. Zhang, and W. Li, “Proactive radio resource optimization with margin prediction: A data mining


approach,” _IEEE Trans. Veh. Technol._, vol. 66, no. 10, pp. 9050–9060, 2017.


[151] K. G. M. Thilina, E. Hossain, and D. I. Kim, “DCCC-MAC: A dynamic common-control-channel-based MAC protocol


for cellular cognitive radio networks,” _IEEE Trans. Veh. Technol._, vol. 65, no. 5, pp. 3597–3613, 2015.


[152] P. Abouzar, K. Shafiee, D. G. Michelson, and V. C. Leung, “Action-based scheduling technique for 802.15. 4/zigbee


wireless body area networks,” in _IEEE International Symposium on Personal, Indoor and Mobile Radio Communications_,


2011, pp. 2188–2192.


[153] J. Wang, C. Jiang, H. Zhang, Y. Ren, K.-C. Chen, and L. Hanzo, “Thirty years of machine learning: the road to pareto

optimal wireless networks,” _IEEE Commun. Surv. Tut._, vol. 22, no. 3, pp. 1472–1514, 2020.


[154] X. Liu, Y. Liu, and Y. Chen, “Reinforcement learning in multiple-UAV networks: Deployment and movement design,”


_IEEE Trans. Veh. Technol._, vol. 68, no. 8, pp. 8036–8049, 2019.


[155] A. Assra, J. Yang, and B. Champagne, “An EM approach for cooperative spectrum sensing in multiantenna CR networks,”


_IEEE Trans. Veh. Technol._, vol. 65, no. 3, pp. 1229–1243, 2015.


[156] A. Morell, A. Correa, M. Barcel´o, and J. L. Vicario, “Data aggregation and principal component analysis in WSNs,”


_IEEE Trans. Wireless Commun._, vol. 15, no. 6, pp. 3908–3919, 2016.


[157] J. Li, H. Zhang, and M. Fan, “Digital self-interference cancellation based on independent component analysis for co-time


co-frequency full-duplex communication systems,” _IEEE Access_, vol. 5, pp. 10 222–10 231, 2017.


[158] S. Niknam, H. S. Dhillon, and J. H. Reed, “Federated learning for wireless communications: Motivation, opportunities,


and challenges,” _IEEE Commun. Mag._, vol. 58, no. 6, pp. 46–51, 2020.


[159] Y. Liu, Z. Qin, M. Elkashlan, Z. Ding, A. Nallanathan, and L. Hanzo, “Non-orthogonal multiple access for 5G and


beyond,” _Proc. IEEE_, vol. 105, no. 12, pp. 2347–2381, 2017.


[160] Y. Li, M. Jiang, Q. Zhang, and J. Qin, “Joint beamforming design in multi-cluster MISO NOMA intelligent reflecting


surface-aided downlink communication networks,” _arXiv:1909.06972_, 2019.


[161] W. Ni, X. Liu, Y. Liu, H. Tian, and Y. Chen, “Resource allocation for multi-cell IRS-aided NOMA networks,” _IEEE_


_Trans. Wireless Commun._, doi: 10.1109/TWC.2021.3057232, 2021.


[162] A. S. d. Sena, D. Carrillo, F. Fang, P. H. J. Nardelli, D. B. d. Costa, U. S. Dias, Z. Ding, C. B. Papadias, and W. Saad, “What


role do intelligent reflecting surfaces play in multi-antenna non-orthogonal multiple access?” _IEEE Wireless Commun._,


vol. 27, no. 5, pp. 24–31, 2020.


[163] X. Yu, D. Xu, and R. Schober, “Enabling secure wireless communications via intelligent reflecting surfaces,” in _IEEE_


_Proc. of Global Commun. Conf. (GLOBECOM)_, 2019, pp. 1–6.


[164] M. Cui, G. Zhang, and R. Zhang, “Secure wireless communication via intelligent reflecting surface,” _IEEE Wireless_


_Commun. Lett._, vol. 8, no. 5, pp. 1410–1414, 2019.


[165] Z. Chu, W. Hao, P. Xiao, and J. Shi, “Intelligent reflecting surface aided multi-antenna secure transmission,” _IEEE Wireless_


_Commun. Lett._, vol. 9, no. 1, pp. 108–112, 2020.


72


[166] L. Dong and H. Wang, “Secure MIMO transmission via intelligent reflecting surface,” _IEEE Wireless Commun. Lett._,


vol. 9, no. 6, pp. 787–790, 2020.


[167] J. Chen, Y. Liang, Y. Pei, and H. Guo, “Intelligent reflecting surface: A programmable wireless environment for physical


layer security,” _IEEE Access_, vol. 7, pp. 82 599–82 612, 2019.


[168] X. Guan, Q. Wu, and R. Zhang, “Intelligent reflecting surface assisted secrecy communication: Is artificial noise helpful


or not?” _IEEE Wireless Commun. Lett._, vol. 9, no. 6, pp. 778–782, 2020.


[169] X. Yu, D. Xu, Y. Sun, D. W. K. Ng, and R. Schober, “Robust and secure wireless communications via intelligent reflecting


surfaces,” _IEEE J. Sel. Areas Commun._, vol. 38, no. 11, pp. 2637–2652, 2020.


[170] S. Goel and R. Negi, “Guaranteeing secrecy using artificial noise,” _IEEE Trans. Wireless Commun._, vol. 7, no. 6, pp.


2180–2189, 2008.


[171] Q. Wang, Z. Chen, H. Li, and S. Li, “Joint power and trajectory design for physical-layer secrecy in the UAV-aided


mobile relaying system,” _IEEE Access_, vol. 6, pp. 62 849–62 855, 2018.


[172] A. Osseiran, F. Boccardi, V. Braun, Kusume _et al._, “Scenarios for 5G mobile and wireless communications: the vision


of the METIS project,” _IEEE Commun. Mag._, vol. 52, no. 5, pp. 26–35, 2014.


[173] Q. Zhang, W. Saad, and M. Bennis, “Reflections in the sky: Millimeter wave communication with UAV-carried intelligent


reflectors,” in _IEEE Proc. of Global Commun. Conf. (GLOBECOM)_, 2019, pp. 1–6.


[174] L. Yang, F. Meng, J. Zhang, M. O. Hasna, and M. Di Renzo, “On the performance of RIS-assisted dual-hop UAV


communication systems,” _IEEE Trans. Veh. Technol._, vol. 69, no. 10, pp. 12 296–12 300, 2020.


[175] X. Mu, Y. Liu, L. Guo, J. Lin, and H. V. Poor, “Intelligent reflecting surface enhanced multi-UAV NOMA networks,”


_arXiv:2101.09145_, 2021.


[176] X. Liu, Y. Liu, and Y. Chen, “Machine learning empowered trajectory and passive beamforming design in UAV-RIS


wireless networks,” _IEEE J. Sel. Areas Commun._, doi:10.1109/JSAC.2020.3041401, 2020.


[177] Q. Liu, J. Wu, P. Xia, S. Zhao, W. Chen, Y. Yang, and L. Hanzo, “Charging unplugged: Will distributed laser charging


for mobile wireless power transfer work?” _IEEE Veh. Technol. Mag._, vol. 11, no. 4, pp. 36–45, 2016.


[178] L. Yao, J. Wang, X. Wang, A. Chen, and Y. Wang, “V2X routing in a VANET based on the hidden Markov model,”


_IEEE Trans. Intell. Transport. Syst_, vol. 19, no. 3, pp. 889–899, 2018.


[179] R. P. D. Vivacqua, M. Bertozzi, P. Cerri, F. N. Martins, and R. F. Vassallo, “Self-localization based on visual lane marking


maps: An accurate low-cost approach for autonomous driving,” _IEEE Trans. Intell. Transport. Syst_, vol. 19, no. 2, pp.


582–597, 2018.


[180] X. Liu, Y. Liu, Y. Chen, and L. Hanzo, “Enhancing the fuel-economy of V2I-assisted autonomous driving: A reinforcement


learning approach,” _IEEE Trans. Veh. Technol._, vol. 69, no. 8, pp. 8329–8342, 2020.


[181] G. Williams, P. Drews, B. Goldfain, J. M. Rehg, and E. A. Theodorou, “Information-theoretic model predictive control:


Theory and applications to autonomous driving,” _IEEE Trans. Robot_, vol. 34, no. 6, pp. 1603–1622, 2018.


[182] A. Makarf, K. M. Rabie, O. Kaiwartya, K. Adhikari, X. Li, M. Quiroz-Castellanos, and R. Kharel, “Reconfigurable


intelligent surfaces-enabled vehicular networks: A physical layer security perspective,” _arXiv:2004.11288_, 2020.


73


[183] J. Wang, W. H. Zhang, X. Bao, T. Song, and C. Pan, “Outage analysis for intelligent reflecting surface assisted vehicular


communication networks,” _arXiv:2005.00996_, 2020.


