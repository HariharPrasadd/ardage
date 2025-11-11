Date of publication xxxx 00, 0000, date of current version xxxx 00, 0000.


_Digital Object Identifier 10.1109/ACCESS.2017.DOI_

## **Scoring the Terabit/s Goal:** **Broadband Connectivity in 6G**


**NANDANA RAJATHEVA** **[1]** **, ITALO ATZENI** **[1]** **, SIMON BICAÏS** **[2]** **, EMIL BJÖRNSON** **[3]** **,**
**ANDRÉ BOURDOUX** **[4]** **, STEFANO BUZZI** **[5]** **, CARMEN D’ANDREA** **[5]** **, JEAN-BAPTISTE DORÉ** **[2]** **,**
**SERHAT ERKUCUK** **[6]** **, MANUEL FUENTES** **[7]** **, KE GUAN** **[8]** **, YUZHOU HU** **[9]** **, XIAOJING HUANG** **[10]** **,**
**JARI HULKKONEN** **[11]** **, JOSEP MIQUEL JORNET** **[12]** **, MARCOS KATZ** **[1]** **, BEHROOZ MAKKI** **[13]** **,**
**RICKARD NILSSON** **[14]** **, ERDAL PANAYIRCI** **[6]** **, KHALED RABIE** **[15]** **,**
**NUWANTHIKA RAJAPAKSHA** **[1]** **, MOHAMMADJAVAD SALEHI** **[1]** **, HADI SARIEDDEEN** **[16]** **,**
**SHAHRIAR SHAHABUDDIN** **[17]** **, TOMMY SVENSSON** **[18]** **, OSKARI TERVO** **[11]** **, ANTTI TÖLLI** **[1]** **,**
**QINGQING WU** **[19]** **, AND WEN XU** **[20]**

1 Centre for Wireless Communications, University of Oulu, Finland (email: {nandana.rajatheva, italo.atzeni, marcos.katz, nuwanthika.rajapaksha,
mohammadjavad.salehi, antti.tolli}@oulu.fi).
2 CEA-Leti, France (email: jean-baptiste.dore@cea.fr).
3 KTH Royal Institute of Technology, Sweden, and Linköping University, Sweden (email: emilbjo@kth.se).
4 IMEC, Belgium (email: andre.bourdoux@imec.be).
5 University of Cassino and Southern Latium, Italy, and Consorzio Nazionale Interuniversitario per le Telecomunicazioni (CNIT), Italy (email: buzzi@unicas.it,
carmen.dandrea@unicas.it).
6 Department of Electrical and Electronics Engineering, Kadir Has University, Turkey (email:{serkucuk, eepanay}@khas.edu.tr).
7 Fivecomm, Valencia, Spain (email: manuel.fuentes@fivecomm.eu).
8 State Key Laboratory of Rail Traffic Control and Safety, Beijing Jiaotong University, China, and with the Beijing Engineering Research Center of High-Speed
Railway Broadband Mobile Communications, China (email: kguan@bjtu.edu.cn).
9 Algorithm Department, ZTE Corporation, China (email: hu.yuzhou@zte.com.cn).
10 School of Electrical and Data Engineering, Faculty of Engineering and Information Technology, University of Technology Sydney, Australia (email:
xiaojing.huang@uts.edu.au).
11 Nokia Bell Labs, Finland (email: {jari.hulkkonen, oskari.tervo}@nokia-bell-labs.com).
12 Institute for the Wireless Internet of Things, Department of Electrical and Computer Engineering, Northeastern University, USA (email:
j.jornet@northeastern.edu).
13 Ericsson Research, Gothenburg, Sweden (email: behrooz.makki@ericsson.com).
14 Department of Computer Science, Electrical and Space Engineering, Luleå University of Technology, Sweden (email: rickard.o.nilsson@ltu.se).
15 Department of Engineering, Manchester Metropolitan University, UK (email: k.rabie@mmu.ac.uk).
16 Division of Computer, Electrical and Mathematical Sciences and Engineering, King Abdullah University of Science and Technology, Saudi Arabia (email:
hadi.sarieddeen@kaust.edu.sa).
17 Nokia, Finland (email: shahriar.shahabuddin@nokia.com).
18 Department of Electrical Engineering, Chalmers University of Technology, Sweden (email: tommy.svensson@chalmers.se).
19 State Key Laboratory of Internet of Things for Smart City and Department of Electrical and Computer Engineering, University of Macau, Macau, China (email:
qingqingwu@um.edu.mo).
20 Huawei Technologies, Germany (email: wen.dr.xu@huawei.com).

Corresponding author: Nandana Rajatheva (e-mail: nandana.rajatheva@oulu.fi).


The work of N. Rajatheva and N. Rajapaksha was supported by the Academy of Finland 6Genesis Flagship (grant 318927). The work of
I. Atzeni was supported by the Marie Skłodowska-Curie Actions (MSCA-IF 897938 DELIGHT). The work of S. Buzzi and C. D’Andrea
was supported by the MIUR Project “Dipartimenti di Eccellenza 2018-2022” and by the MIUR PRIN 2017 Project “LiquidEdge”. The
work of E. Panayirci was supported by the Scientific and Technical Research Council of Turkey (TUBITAK) under the 1003-Priority Areas
R&D Projects Support Program No. 218E034.


**ABSTRACT** This paper explores the road to vastly improving the broadband connectivity in future 6G
wireless systems. Different categories of use cases are considered, with peak data rates up to 1 Tbps. Several
categories of enablers at the infrastructure, spectrum, and protocol/algorithmic levels are required to realize
the intended broadband connectivity goals in 6G. At the infrastructure level, we consider ultra-massive
MIMO technology (possibly implemented using holographic radio), intelligent reflecting surfaces, usercentric cell-free networking, integrated access and backhaul, and integrated space and terrestrial networks.
At the spectrum level, the network must seamlessly utilize sub-6 GHz bands for coverage and spatial
multiplexing of many devices, while higher bands will be mainly used for pushing the peak rates of pointto-point links. Finally, at the protocol/algorithmic level, the enablers include improved coding, modulation,
and waveforms to achieve lower latency, higher reliability, and reduced complexity.


**INDEX TERMS** 6G, cell-free massive MIMO, holographic MIMO, integrated access and backhaul,
intelligent reflecting surfaces, Terahertz communications, visible light communications.


VOLUME 4, 2021 1


**I. INTRODUCTION**
HILE fifth-generation (5G) wireless networks are
# W being deployed in many parts of the world and 3rd

generation partnership project (3GPP) is about to freeze
the new long-term evolution (LTE) Release 16, providing
enhancements to the first 5G new radio (NR) specifications
published in 2018, the research community is starting to
investigate the next generation of wireless networks, which
will be designed during this decade and will shape the development of the society during the next decade. The following
are key questions: _(a)_ What will beyond-5G (B5G) and the
sixth-generation (6G) wireless networks look like? _(b)_ Which
new technology components will be at their heart? (c) Will
the 1 Tbps frontier be reached in practice? (d) Will uniform
coverage and quality of service be fully achieved?
At the moment, it is very challenging and risky to provide
definite answers to these questions, even though some key
concepts and statements can be already made. A conservative
and cautious answer is that _6G networks will be based on_

_a combination of 5G with other known technologies that_
_are not mature enough for being included in 5G_ . Based
on this argument, any technology that will not be in the
3GPP standards by the end of 2020 will be a possible ingredient of future generations of wireless cellular systems.
As an example, full massive multiple-input multiple-output
(MIMO) with digital signal-space beamforming is one technology that, although known for some years now, has not
been fully embraced by equipment manufacturers in favor of
the more traditional codebook-based beamforming. A more
daring answer is that _6G networks will be based on new_
_technologies that were not at all considered when design-_
_ing and developing 5G combined with vast enhancements_
_of technologies that were already present in the previous_
_generation of wireless cellular networks_ . As an example, the
use of advanced massive MIMO schemes together with cellfree and user-centric network deployments will combine an
advanced version of a 5G technology (i.e., massive MIMO),
with the fresh concept of cell-free network architectures.
Certainly, we can state that 6G wireless systems will:

_•_ Be based on **extreme densification of the network**
**infrastructure**, such as access points (APs) and intelligent reflecting surfaces (IRSs), which will cooperate
to form a cell-free network with seamless quality of
service over the coverage area.

_•_ Make intense use of **distributed processing and cache**
**memories**, e.g., in the form of cloud-RAN technology.

_•_ Continue the trend of complementing the wide-area
coverage achieved at sub-6 GHz frequencies by using
**substantially higher carrier frequencies beyond the**
**mmWave through the Terahertz (THz) band and up**
**to visible light (VL)** to provide high-capacity point-topoint links.

_•_ Leverage **network slicing and multi-access edge com-**
**puting** to enable the birth of new services with specialized performance requirements and to provide the
needed resources to support vertical markets.



N. Rajatheva _et al._ : Scoring the Terabit/s Goal: Broadband Connectivity in 6G


_•_ Witness an increasing **integration of terrestrial and**
**satellite wireless networks**, with a big role played by
unmanned aerial vehicles (UAVs) and low-Earth orbit
(LEO) micro satellites, to fill coverage holes and offload
the network in heavy-load situations.

_•_ Leverage **machine learning methodologies** to improve
the efficiency of traditional model-based algorithms for
signal processing and resource allocation.
The goal of this paper is to provide a vast and thorough
review of the main technologies that will permit enhancing
the broadband connectivity capabilities of future wireless
networks. The paper indeed extends the huge collaborative
effort that has led to the white paper [1] with a more specific
focus on broadband connectivity, which might be the most
important use case in 6G (though far from the only one)
and with abundant technical details and numerical study
examples. The paper describes the physical-layer (PHY)
and medium-access control (MAC) layer methodologies for
achieving 6G broadband connectivity with very high data
rates, up to the Tbps range.


_**A. RELATED WORKS**_

The existing literature contains several visionary articles that
speculate around the potential 6G requirements, architecture,
and key technologies [2]–[6]. One of the earliest articles
with a 6G vision and related requirements is [2], which
analyzed the need of 6G from a user’s perspective and
concluded that ultralong battery lifetime (maximum energy
savings) will be the key focus of 6G rather than high bit
rates. The authors envisioned that indoor communications

will be completely changed by moving away from wireless
radio communications to optical free-space communications.
In [7], the authors presented a vision for 6G mobile networks
and an overview of the latest research activities on promising
techniques that might be available for utilization in 6G. This
work claimed that the major feature of 6G networks will
be their flexibility and versatility, and that the design of 6G
will require a multidisciplinary approach. New technology
aspects that might evolve wireless networks towards 6G are
also discussed in [3]. The authors analyzed the scenarios and
requirements associated to 6G networks from a full-stack
perspective. The technologies aiming to satisfy such requirements are discussed in terms of spectrum usage, PHY/MAC
and higher layers, network architectures, and intelligence
for 6G. In [5], the authors highlighted new services and
core enabling technologies for 6G networks, identifying subTHz and optical communication as the key physical layer
technologies to achieve the 6G requirements. The authors
also provided a 6G roadmap that starts with forming a 6G
vision and ends in a 6G proof-of-concept evaluation using
testbeds.

A vision for 6G that could serve as a research guide
for the post-5G era was presented in [8]. Therein, it was
suggested that high security, secrecy, and privacy will be the
key features of 6G because human-centric communications
will still be the most important application. The authors



2 VOLUME 4, 2021


N. Rajatheva _et al._ : Scoring the Terabit/s Goal: Broadband Connectivity in 6G


provided a framework to support this vision and discussed
the potential application scenarios of 6G; furthermore, they
examined the pros and cons of candidate technologies that
recently appeared. In [9], the communication needs and technologies in the time frame of 6G were presented. According
to the authors, the future of connectivity will lie in unifying
our experience across true representation of the physical and
biological worlds at every spatial point and time instant. New
themes such as man-machine interfaces, ubiquitous universal
computing, multi-sensory data fusion, and precision sensing
and actuation will drive the 6G system requirements and
technologies. The authors predicted that artificial intelligence
(AI) has the potential to become the foundation for the 6G air
interface and discussed five more major technology transformations to define 6G. An analysis of 6G networks, extending
to its relationships with B5G, core services, required key performance indicators (KPIs), enabling technologies, architecture, potential challenges, promising solutions, opportunities,
and future developmental direction, was presented in [10].
The authors identified five core services as well as eight KPIs
for 6G.

Networld2020 is a European Technology Platform for
community discussion on the future research technologies
in telecommunications. A white paper on the strategic research and innovation agenda for 2021–27 was published in
2018 and updated in 2020 by a large group of researchers
in order to provide guidance for the development of the
future European Union R&D program, such as the 9-th EU
Framework Program [11]. In particular, the chapter “Radio
Technology and Signal Processing” of the SRIA listed ten
enabling technologies relevant to the 6G air interface design.
Some prospective key enabling techniques for 6G and beyond have been surveyed in [12]. The authors envisioned
that the 6G wireless systems will be mainly driven by a
focus of unrestricted availability of high quality wireless
access. The authors highlighted technologies that are vital to
the success of 6G and discussed promising early-stage technologies for beyond 6G communications such as Internet of
NanoThings, the Internet of BioNanoThings, and Quantum
Communications. Lastly, [13] envisioned that large antenna
arrays will be omnipresent in B5G. This work described how
the technology can evolve to include larger, denser, and more
distributed arrays, which can be used for communication,
positioning, and sensing.


_**B. PAPER CONTRIBUTION AND ORGANIZATION**_

This paper differs from previously cited related works on 6G
papers for three main reasons: (i) it is not a general paper
on 6G network but it has a special focus on the broadband
connectivity, i.e. it describes the key technologies at the PHY
and MAC layers, and the related research challenges, that
will permit achieving broadband wireless connections at 1
Tbps in multiuser scenarios; (ii) the paper length and exhaustiveness makes it different from previously cited papers that
have appeared in magazines and not in technical journals; and
(iii) differently from many review papers that turn down the



use of equations and illustrate concepts in a colloquial style,
this paper contains several technical dives into various key
topics, providing in-depth details, equations, and illustrative
numerical results to corroborate the technical exposition.
The rest of this paper is organized as follows. Section II
provides a discussion about the technologies adopted and
included in current 5G NR specifications. This helps shedding some light on the discrepancy between technologies and
algorithms available in the scientific literature and the ones
that are really retained by 3GPP for inclusion in their released
specifications; moreover, it also helps to realize that some
of the technologies that almost a decade ago were claimed
to be part of future 5G systems now are claimed as being
part of future 6G networks. An illustrative example of this
phenomenon is the use of carrier frequencies above-30 GHz:
for some reason, technology at such high frequencies was
not yet mature for mass market production at a convenient
cost, and so the massive use of higher frequencies has been
delayed and is not currently part of 5G NR specifications.
Section II also provides a discussion of what the authors
believe to be the most relevant use cases for future 6G

networks. The detailed use cases describe applications and
scenarios that cannot be implemented with current 5G technology, and that need a boost in broadband connectivity of
2-3 orders of magnitude. Finally, the section is concluded
with a description of KPI values envisioned for the next
generation of mobile networks. Sections III–V are the containers of the vast bulk of technologies that will enable Tbps
wireless rates in 6G networks. We define these technologies
as “enablers” for achieving 6G broadband connectivity, as
illustrated in Fig. 1, and categorize them into enablers at
the spectrum level (Section III), at the infrastructure level
(Section IV), and at the protocol/algorithmic level (Section
V). Section III contains thus a treatment of the enablers at
the spectrum level. These include the use of THz, optical
and VL carrier frequencies. The most promising bandwidths
are discussed, along with their pros and cons. In particular, the section provides a detailed discussion of the challenges posed by the propagation channel at THz frequencies.
Section IV is devoted to the enablers at the infrastructure

level, and thus treats all the technological innovations that
mobile operators will have to deploy in order to support
Tbps connectivity. These include ultra-massive MIMO and
holographic radio, IRSs, scalable cell-free networking with
user-centric association, the use of integrated access and
backhaul, the integration of terrestrial and non-terrestrial
networks, and wideband broadcasting. The Section provides
in-depth technical discussion, with equations and numerical
results, about the advantages granted by the use of IRSs,
by the adoption of cell-free network deployment in place
of traditional massive MIMO networks, and by the use of
integrated access and backhaul technologies. Then, Section V
surveys the enablers at the protocol/algorithmic level. These
include new coding, modulation, and duplexing schemes,
new transceiver algorithms based on machine learning and
coded caching. The Section provides illustrative numerical



VOLUME 4, 2021 3


N. Rajatheva _et al._ : Scoring the Terabit/s Goal: Broadband Connectivity in 6G











































**FIGURE 1.** The expected enablers for achieving 6G broadband access.



results about the performance of polar-QAM constellations
for THz communications, and of the spatial tuning methodology to design antenna arrays, still for THz communications.
Numerical results are presented also to show the potential of
the autoencoder methodology in the design of data detection
algorithms based on machine learning. Section VI gives an
overview of the technologies to provide full coverage in rural
and underdeveloped areas. Indeed, one of the challenges of
the wireless communications community for the next decade
is to extend the benefits of broadband connections to the
largest possible share of the world population. As discussed
in the section, full coverage will be possible through a proper
exploitation and combination of some of the methodologies
introduced in Sections III-V. Finally, Section VII contains
concluding remarks and wraps up the paper. For the reader’s
ease, the list of acronyms used in this paper is reported in
Table I-B.


**II. FROM 5G TO 6G**

This section describes the state-of-the-art in 5G and presents
our vision for 6G use cases and performance indicators.


_**A. 5G TODAY**_

The 5G NR sets the foundations of what is 5G today. In terms
of modulation, 5G uses the same waveforms as in 4G, except
that orthogonal frequency-division multiplexing (OFDM) is



mandatory now both in downlink and uplink. Moreover, discrete Fourier transform-spread-OFDM (DFT-s-OFDM) can
be used in the uplink. This is a single carrier (SC) like transmission scheme mainly designed for coverage-limited cases
due to its better power efficiency. The same waveforms are
envisioned to be used up to the 71 GHz frequency band. At
the moment, the maximum supported number of transmitted
sub-carriers is 3168 which corresponds to a 400 MHz channel
bandwidth using 120 kHz subcarrier spacing [14] (or up to
100 MHz in sub-6 GHz), which can be further aggregated to
up to 800 MHz bandwidth. The current study for extending
the NR specification up to 71 GHz will consider channel
bandwidths up to 2 GHz. The 5G NR standard utilizes new
channel coding schemes compared to earlier generations. The
standard uses low-density parity-check (LDPC) coding for
the data channels, while polar coding is used for the control
channels when having more than 11 payload bits. The bits
are mapped to modulation symbols using either QPSK, 16QAM, 64-QAM and 256-QAM both in downlink and uplink,
while _π/_ 2-BPSK is supported for the DFT-s-OFDM uplink,
which can enable extreme coverage due to its low peak-toaverage power ratio [15]. For _π/_ 2-BPSK, the user equipment
(UE) is allowed to use very power-efficient transmission by
employing frequency-domain spectral shaping.


Multi-antenna techniques are already one of the key parts
of the current 5G NR specification. Especially in the lower



4 VOLUME 4, 2021


N. Rajatheva _et al._ : Scoring the Terabit/s Goal: Broadband Connectivity in 6G


**TABLE 1.** List of Acronyms

|Acronym|Definition|Acronym|Definition|
|---|---|---|---|
|~~3GPP~~<br>|~~3rd generation partnership project~~<br>|~~ITU-R~~<br>|~~ITU Radiocommunication Sector~~<br>|
|~~5G~~<br>|~~Fifth-generation~~<br>|~~KPI~~<br>|~~Key performance indicator~~<br>|
|~~6G~~<br>|~~Sixth-generation~~<br>|~~LDPC~~<br>|~~Low-density parity-check~~<br>|
|~~AI~~<br>|~~Artiﬁcial intelligence~~<br>|~~LED~~<br>|~~Light emitting diode~~<br>|
|~~AoA~~<br>|~~Angle of arrival~~<br>|~~LEO~~<br>|~~Low-Earth orbit~~<br>|
|~~AoD~~<br>|~~Angle of departure~~<br>|~~LiFi~~<br>|~~Light ﬁdelity~~<br>|
|~~AP~~<br>|~~Access point~~<br>|~~LIoT~~<br>|~~Light-based Internet of things~~<br>|
|~~APSK~~<br>|~~Amplitude and phase-shift keying~~<br>|~~LoS~~<br>|~~Line-of-sight~~<br>|
|~~AR~~<br>|~~Augmented reality~~<br>|~~LTE~~<br>|~~Long-term evolution~~<br>|
|~~ARQ~~<br>|~~Automatic repeat request~~<br>|~~M2M~~<br>|~~Machine-to-machine~~<br>|
|~~AWGN~~<br>|~~Additive white Gaussian noise~~<br>|~~MAC~~<br>|~~Medium access control~~<br>|
|~~B5G~~<br>|~~Beyond 5G~~<br>|~~MBS~~<br>|~~Macro base station~~<br>|
|~~BER~~<br>|~~Bit error rate~~<br>|~~MR~~<br>|~~Mixed reality~~<br>|
|~~BiCMOS~~<br>|~~Bipolar complementary metal-oxide-semiconductor~~<br>|~~MEO~~<br>|~~Medium-Earth orbit~~<br>|
|~~BP~~<br>|~~Belief propagation~~<br>|~~MIMO~~<br>|~~Multiple-input multiple-output~~<br>|
|~~BPSK~~<br>|~~Binary phase-shift keying~~<br>|~~ML~~<br>|~~Machine learning~~<br>|
|~~BS~~<br>|~~Base station~~<br>|~~mMIMO~~<br>|~~Massive multiple-input multiple-output~~<br>|
|~~CA-CSL~~<br>|~~CRC-aided successive cancellation list~~<br>|~~MMSE~~<br>|~~Minimum mean square error~~<br>|
|~~CC~~<br>|~~Coded caching~~<br>|~~mmWave~~<br>|~~Millimeter wave~~<br>|
|~~CDF~~<br>|~~Cumulative distribution function~~<br>|~~MS~~<br>|~~Multi-antenna scheme~~<br>|
|~~CF-UC~~<br>|~~Cell-free with user-centric~~<br>|~~NET~~<br>|~~Network layer~~<br>|
|~~CMOS~~<br>|~~Complementary metal-oxide-semiconductor~~<br>|~~NGI~~<br>|~~Next generation of internet~~<br>|
|~~CoMP~~<br>|~~Coordinated multi-point~~<br>|~~NLoS~~<br>|~~Non-line-of-sight~~<br>|
|~~CPRI~~<br>|~~Common public radio interface~~<br>|~~NOMA~~<br>|~~Non-orthogonal multiple access~~<br>|
|~~CPU~~<br>|~~Central processing unit~~<br>|~~OFDM~~<br>|~~Orthogonal frequency-division multiplexing~~<br>|
|~~CRC~~<br>|~~Cyclic redundancy check~~<br>|~~NR~~<br>|~~New radio~~<br>|
|~~CSI~~<br>|~~Channel state information~~<br>|~~OPEX~~<br>|~~Operating expense~~<br>|
|~~DFT~~<br>|~~Discrete Fourier transform~~<br>|~~OWC~~<br>|~~Optical wireless communications~~<br>|
|~~DFT-s-OFDM~~<br>|~~DFT-spread-orthogonal frequency-division multiplexing~~<br>|~~PAPR~~<br>|~~Peak-to-average-power-ratio~~<br>|
|~~DMRS~~<br>|~~Demodulation reference signals~~<br>|~~PDF~~<br>|~~Probability density function~~<br>|
|~~EoA~~<br>|~~Elevation angle of arrival~~<br>|~~PDP~~<br>|~~Power delay proﬁle~~<br>|
|~~EoD~~<br>|~~Elevation angle of departure~~<br>|~~PER~~<br>|~~Packet error rate~~<br>|
|~~ETSI~~<br>|~~European Telecommunications Standards Institute~~<br>|~~PHY~~<br>|~~Physical layer~~<br>|
|~~FCC~~<br>|~~Federal Communications Commission~~<br>|~~PN~~<br>|~~phase noise~~<br>|
|~~FCF~~<br>|~~Full cell-free~~<br>|~~PPA~~<br>|~~Proportional power allocation~~<br>|
|~~FDD~~<br>|~~Frequency division duplex~~<br>|~~PPA-DL~~<br>|~~Proportional power allocation-dowlink~~<br>|
|~~FEC~~<br>|~~Forward error correction~~<br>|~~QAM~~<br>|~~Quadrature amplitude modulation~~<br>|
|~~FHPPP~~<br>|~~Finite homogeneous Poisson point processes~~<br>|~~QPSK~~<br>|~~Quadrature phase-shift keying~~<br>|
|~~FoV~~<br>|~~Field-of-view~~<br>|~~RED~~<br>|~~Reduced-subpacketization scheme~~<br>|
|~~FPA-DL~~<br>|~~Fractional power allocation-dowlink~~<br>|~~RF~~<br>|~~Radio frequency~~<br>|
|~~FPA-UL~~<br>|~~Fractional power allocation-uplink~~<br>|~~RS~~<br>|~~Rate splitting~~<br>|
|~~FSO~~<br>|~~Free-space optics~~<br>|~~RT~~<br>|~~Ray tracing~~<br>|
|~~GEO~~<br>|~~Geostationary-Earth orbit~~<br>|~~SBS~~<br>|~~Small base station~~<br>|
|~~HAPS~~<br>|~~High-altitude platform station~~<br>|~~SC~~<br>|~~Single-carrier~~<br>|
|~~HPC~~<br>|~~High performance computing~~<br>|~~SNR~~<br>|~~Signal-to-noise ratio~~<br>|
|~~HST~~<br>|~~High-speed train~~<br>|~~SWIPT~~<br>|~~Simultaneous wireless information and power transfer~~<br>|
|~~IAB~~<br>|~~Integrated access and backhaul~~<br>|~~TDD~~<br>|~~Time division duplex~~<br>|
|~~IBFD~~<br>|~~In-band full-duplex~~<br>|~~UAV~~<br>|~~Unmanned aerial vehicle~~<br>|
|~~I/O~~<br>|~~Input/output~~<br>|~~UE~~<br>|~~User equipment~~<br>|
|~~IEEE~~<br>|~~Institute of Electrical and Electronics Engineers~~<br>|~~UPA-DL~~<br>|~~Uniform power allocation-downlink~~<br>|
|~~IID~~<br>|~~Independent and identically distributed~~<br>|~~UPA-UL~~<br>|~~Uniform power allocation-uplink~~<br>|
|~~IMT~~<br>|~~International Mobile Telecommunications~~<br>|~~UV~~<br>|~~Ultraviolet~~<br>|
|~~IoT~~<br>|~~Internet of things~~<br>|~~UW~~<br>|~~Ultra-wideband~~<br>|
|~~IR~~<br>|~~Infrared~~<br>|~~V2X~~<br>|~~Vehicle-to-everything~~<br>|
|~~IRS~~<br>|~~Intelligent reﬂecting surface~~<br>|~~VL~~<br>|~~Visible light~~<br>|
|~~ISTN~~<br>|~~Integrated space and terrestrial network~~<br>|~~VLC~~<br>|~~Visible light communications~~<br>|
|~~ITS~~<br>|~~Intelligent transportation systems~~<br>|~~VR~~<br>|~~Virtual reality~~<br>|
|~~ITU~~|~~International Telecommunication Union~~|~~WLAN~~|~~Wireless local area network~~|



frequencies, MIMO technology is used to provide spatial
multiplexing gains, either by multiplexing multiple users
(multi-user MIMO) or increasing the throughput of a single
user (single-user MIMO). At the moment, in the downlink, a
maximum of two codewords mapped to a maximum of eight



layers for a single UE and a maximum of four (orthogonal)
layers per UE for multi-user MIMO are supported [16].
Basically, a maximum of 12 orthogonal DMRS ports are
supported, but if the transmitter can perform some form of
orthogonalization between the layers, such as using spatial



VOLUME 4, 2021 5


separation, an even higher number of layers is possible [15].
In the uplink, a maximum of four layers are supported.
Single-user MIMO is not supported for DFT-s-OFDM. In
the uplink, both codebook and non-codebook based transmissions are supported. In the codebook based precoding,
the gNB informs the UE of the transmit precoding matrix
to use in the uplink, while in the non-codebook based precoding, the UE can decide the precoding based on feedback
from its sounding reference signal. However, only wideband
precoding is supported in both cases, which means that the
same precoding must be utilized over all allocated subcarriers. While academia is often regarding 64 antenna ports
has the minimum number that constitutes massive MIMO

[13], 3GPP defines massive MIMO as having more than
eight ports, so massive MIMO is already widely supported.
However, there have been plenty of commercial products
with both 32 and 64 ports for use in sub-6 GHz bands [13].
In the higher frequency bands, the number of antenna
elements increase significantly to compensate for the element
size shrinks and increasing signal penetration losses. 5G
NR in mmWave frequencies has been designed relying on
the assumption of many antenna elements which perform
analog beamforming. Basically the beam sweeping phase
in sub-6 GHz frequencies supports up to 8 beams in one
dimension, while up to 64 different beams can be used in
two dimensions (vertical/horizontal) in higher bands [17].
Release 16 has also included support for transmissions from
multiple transmission and reception points and several other
enhancements to facilitate use of large number of antennas,
and the improvements will further continue in Release 17

[18].
There are new features in Release 16 such as 5G in

unlicensed bands, operations using multiple transmission and
reception points, V2X applications, or IAB. The latter is a
solution to provide backhaul or relay connectivity using the
same resources as for UE access, without requiring additional
sites, equipment, or resources. The donor gNB has a fiber
connection to the core network. The radio connection from

the donor gNB to the IAB node is the same as used to connect
UEs [19].
The work on Release 17 will start in the second half

of 2020. It is expected to include enhancements of the
broadband connectivity related to MIMO, dynamic spectrum
sharing, coverage extension, dual connectivity, UE power
saving, IAB, and data collection. As new features, there
will be support for up to 71 GHz frequencies, multicast and
broadcast services [20], multi-SIM devices, non-terrestrial
networks, and sidelink relaying.


_**B. 6G USE CASES**_

The emergence and need for 6G technology will be governed
by unprecedented performance requirements arising from
exciting new applications foreseen in the 2030 era, which
existing cellular generations will not be able to support. This
paper focuses on applications that require broadband connectivity with high data rates and availability, in combination



N. Rajatheva _et al._ : Scoring the Terabit/s Goal: Broadband Connectivity in 6G


with other specialized characteristics. The following is a list
of the potential new use cases and applications in 6G which
will help to understand the key requirements of future 6G
systems. The list should not be seen as a replacement but
complement to existing 5G use cases. A summary of the use
cases discussed is presented in Fig. 2.


_•_ **Extreme capacity xHaul:** This use case refers to a fixed
symmetric point-to-point link targeting high data rates
without energy or complexity constraints since no user
devices are involved. This can only be enabled using a
combination of high bandwidth and high spectral efficiency. The envisioned ultra-dense network topology in
urban areas with extreme capacity and latency requirements makes fiber-based backhauling highly desirable
but complicated due to the limited fiber network penetration (variable from country to country) and related
extension costs. Hence, wireless infrastructure is needed
as a flexible means to complement optical fiber deployment, both indoors and outdoors, to avoid bottlenecks
in the backhaul (or xHaul). Ultra-high speed is required
since the backhaul aggregates the data rates of many
user devices. The xHaul can also provide efficient access
to computing resources at the edge or in the cloud. Fig. 3
depicts an example setup of an extreme capacity xHaul
network in an indoor environment.


_•_ **Enhanced hotspot:** An enhanced hotspot entails a highrate downlink from the AP to several user devices, with
short coverage and low receiver complexity constraints.
The envisaged applications are hotspots delivering highspeed data to demanding applications such as highdefinition video streaming and enhanced WLAN technology.

_•_ **Short-range device-to-device communications:** A
symmetric high-rate point-to-point link with very stringent energy and complexity constraints is considered
here. This use case focuses on data exchange between
user devices that are within a short distance, with lim
ited involvement of the network infrastructure. This use

case also includes inter/intra-chip communications and
wireless connectors, among others.

_•_ **High-mobility hotspot:** This use case is an umbrella
for multiple applications, such as smart bus, rail, underground, or even airplane connectivity. High-speed mobile communication while on mass public transportation
systems has drastically evolved in the last few years
and it is expected by passengers. As part of these applications, there are different scenarios to be considered,
including user to hotspot connectivity (e.g., inside the
train/bus/plane), mobile hotspot to fixed infrastructure
(e.g., from trains/bus/plane to fixed networked stations),
moving base station (providing access on the move to
users outside the vehicle), or hotspot to hotspot, among
others. The biggest challenge is posed by the mobile
hotspot to fixed infrastructure scenario because this link
needs to achieve very high data rates, low latencies,



6 VOLUME 4, 2021


N. Rajatheva _et al._ : Scoring the Terabit/s Goal: Broadband Connectivity in 6G


**High-Mobility Hotspot**


**• Smart bus, rail, underground and**
**airplane connectivity**

**• Interconnected users, hotspots**
**and infrastructure**

**• Very high data rates, low**
**latencies, and 100% availability at**
**very high speeds**


**FIGURE 2.** Different 6G use cases and applications.



as well as close to 100% availability while traveling
at speeds potentially exceeding hundreds of km/h. To
support all users’ aggregated traffic, a bandwidth of
several GHz (or even higher) is needed to accommodate
up to 100 Gbps data rates. Such high-data rate and
huge bandwidth requirements are a strong motivation to
explore the mmWave and THz bands [22].

_•_ **Multi-sensory extended reality:** AR/MR/VR applications, capturing multi-sensory inputs and providing realtime user interaction are considered under this use case.

Extremely high per-user data rates in the Gbps range
and exceptionally low latencies are required to deliver
a fully immersive experience [3]. Remote connectivity
and interaction powered by holographic communications, along with all human sensory input information,
will further push the data rate and latency targets.
Multiple-view cameras used for holographic communications will require data rates in the order of terabits per
second [5].

_•_ **Industrial automation and robotics:** Industry 4.0 envisions a digital transformation of manufacturing industries and processes through cyber-physical systems, IoT
networks, cloud computing, and artificial intelligence.
In order to achieve high-precision manufacturing, automatic control systems, and communication technologies are utilized in the industrial processes. Ultra-high
reliability in the order of 1-10 _[−]_ [9] and extremely low
latency around 0.1-1 ms round-trip time are expected in
communications, along with real-time data transfer with
guaranteed microsecond delay jitter in industrial control



networks [5]. While 5G initiated the implementation of
Industry 4.0, 6G is expected to reveal its full potential
supporting these stringent requirements by using novel,
disruptive technologies brought by 6G.

_•_ **Autonomous mobility:** The smart transportation technologies initiated in 5G are envisioned to be further
improved towards fully autonomous systems, providing
safer and more efficient transportation, efficient traffic management, and improved user experiences. Connected autonomous vehicles demand reliability above
99.99999% and latency below 1 ms, even in very high
mobility scenarios up to 1000 km/h [3], [5]. Moreover,
higher data rates are required due to the increased
number of sensors in vehicles that are needed to assist

autonomous driving. Other autonomous mobility solutions such as drone-delivery systems and drone swarms
are also evolving in different application areas such
as construction, emergency response, military, etc. and
require improved capacity and coverage [3].

_•_ **Connectivity in remote areas:** Half of the world’s population still lacks basic access to broadband connectiv
ity. The combination of current technologies and business models have failed to reach large parts of the world.
To reduce this digital divide, a key target of 6G is to
guarantee 10 Mbps in every populated area of the world,
using a combination of ground-based and spaceborne
network components. Importantly, this should not only
be theoretically supported by the technology but 6G
must be designed in a sufficiently cost-efficient manner
to enable actual deployments that deliver broadband to



VOLUME 4, 2021 7


N. Rajatheva _et al._ : Scoring the Terabit/s Goal: Broadband Connectivity in 6G


**FIGURE 3.** An example setup of an extreme capacity xHaul network in an indoor environment. Source: [21].



the entire population of the world.

_•_ **Other use cases:** Some other applications that 6G is
expected to enable or vastly enhance include: internet access on planes, wireless brain-computer interface
based applications, broadband wireless connectivity inside data centers, Internet of Nano-Things and Internet
of Bodies through smart wearable devices and intrabody
communications achieved by implantable nanodevices
and nanosensors [4].


_**C. 6G KEY PERFORMANCE INDICATORS**_


The KPIs have to a large extent stayed the same for several network generations [23], [24], while the minimum
requirements have become orders-of-magnitude sharper. One
exception is the energy efficiency, which was first introduced
as a KPI in 5G, but without specifying concrete targets. We
believe 6G will mainly contain the same KPIs as previous
generations but with much higher ambitions. However, it is of
great importance that the existing KPIs be critically reviewed
and new KPIs be seriously considered covering both technology and productivity related aspects and sustainability and
societal driven aspects [1]. While the KPIs were mostly independent in 5G (but less stringent at high mobility and over
large coverage areas), a cross-relationship is envisaged in 6G
through a definition of groups. All the indicators in a group
should be fulfilled at the same time, but different groups can



**TABLE 2.** A Comparison of 5G and 6G KPIs [3]–[5], [25]

|KPIs|5G|6G|
|---|---|---|
|~~Peak data rate~~<br>|20~~ Gb/s~~<br>|1~~ Tb/s~~<br>|
|~~Experienced data rate~~<br>|0_._1~~ Gb/s~~<br>|1~~ Gb/s~~<br>|
|~~Peak spectral efﬁciency~~<br>|30~~ b/s/Hz~~<br>|60~~ b/s/Hz~~<br>|
|~~Experienced spectral efﬁ-~~<br>ciency<br>|0_._3~~ b/s/Hz~~<br>|3~~ b/s/Hz~~<br>|
|~~Maximum bandwidth~~<br>|1~~ GHz~~<br>|100~~ GHz~~<br>|
|~~Area trafﬁc capacity~~<br>|10~~ Mb/s/m2~~<br>|1~~ Gb/s/m2~~<br><br>|
|~~Connection density~~<br>|10~~6 devices/~~<br>|~~km2~~<br>10~~7 devices/km2~~<br><br>|
|~~Energy efﬁciency~~<br>|~~Not speciﬁe~~<br>|~~d~~<br>1~~ Tb/J~~<br>|
|~~Latency~~<br>|1~~ ms~~<br>|100_ µ_~~s~~<br>|
|~~Reliability~~<br>|1_ −_10~~_−_5~~<br>|1_ −_10~~_−_9~~<br><br>|
|~~Jitter~~<br>|~~Not speciﬁe~~<br>|~~d~~<br>1_ µ_~~s~~<br>|
|~~Mobility~~|500~~ km/h~~|1000~~ km/h~~|



have diverse requirements. The reason for this is that we will
move from a situation where broadband connectivity is delivered in a single way to a situation where the requirements of
different broadband applications will become so specialized
that their union cannot be simultaneously achieved. Hence,
6G will need to be configurable in real-time to cater to these
different groups.
The following are the envisaged KPIs.


_•_ _Extreme data rates_ : Peak data rates up to 1 Tb/s are
envisaged for both indoor and outdoor connectivity. The
user-experienced data rate, which is guaranteed to 95%



8 VOLUME 4, 2021


N. Rajatheva _et al._ : Scoring the Terabit/s Goal: Broadband Connectivity in 6G


of the user locations, is envisioned to reach 1 Gb/s.

_•_ _Enhanced spectral efficiency and coverage_ : The peak
spectral efficiency can be increased using improved
MIMO technology and modulation schemes, likely up
to 60 b/s/Hz. However, the largest envisaged improvements are in terms of the uniformity of the spectral
efficiency over the coverage area. The user-experienced
spectral efficiency is envisaged to reach 3 b/s/Hz. Moreover, new physical layer techniques are needed to allow
for broadband connectivity in high mobility scenarios
and more broadly in scenarios for which former wireless
networks generations do not fully meet the needs.

_•_ _Very wide bandwidths_ : To support extremely high peak
rates, the maximum supported bandwidth must greatly
increase. Bandwidths up to 10 GHz can be supported
in mmWave bands, while more than 100 GHz can be
reached in sub-THz, THz, and VL bands.

_•_ _Enhanced energy efficiency_ : Focusing on sustainable development, 6G technologies are expected to pay special
attention in achieving better energy efficiency, both in
terms of the absolute power consumption per device
and the transmission efficiency. In the latter case, the
efficiency should reach up to 1 terabit per Joule. Hence,
developing energy-efficient communication strategies is
a core component of 6G.

_•_ _Ultra-low latency_ : The use of bandwidths that are wider
than 10 GHz will allow for latency down to 0.1 ms. The
latency variations (jitter) should reach down to 1 _µ_ s, to
provide an extreme level of determinism.

_•_ _Extremely high reliability_ : Some new use cases require
extremely high reliability up to 1-10 _[−]_ [9] to enable mission and safety-critical applications.
It is unlikely that all of these requirements will be simultaneously supported, but different use cases will have different
sets of KPIs, and only some will have the maximum requirements mentioned above. A comparison of the 5G and 6G
KPIs is shown in Table 2, where also the area traffic capacity
and connection density are considered. The KPIs presented
there are not limited only to the broadband scenario, but
provides an overall idea of expected KPIs for 6G in a broader
view. More detailed KPIs for different verticals and use cases

can be found in [26] and [27].
It is highly likely that 6G will to a large extent carry information related also to non-traditional applications of wireless
communications, such as distributed caching, computing, and
AI decisions. Thus, we need to investigate whether there is a
need to introduce new KPIs for such applications, or if the
traditional KPIs are sufficient.


1) Scoring the Terabit/s goal
As indicated by the title of this paper, our main focus is to
identify different technologies that can be utilized to reach
1 Tb/s. Such a high KPI requirement can appear in pointto-point use cases, between a single transmitter and a single
receiver, for extreme capacity xHaul and device-to-device
communications. Alternatively, 1 Tb/s can be the accumulate



capacity requirement in point-to-multipoint use cases, for example, hotspots where a base station is spatially multiplexing
a larger number of devices. To give an indication of how one
can reach 1 Tb/s, the following high-level capacity formula
can be utilized:


Multiplexing gain _×_ Bandwidth _×_ Spectral efficiency

= 1 Tb _/_ s _._ (1)


Hence, there are three multiplicative factors that we can
play with: 1) The multiplexing gain, which represents the
number of devices that are spatially multiplexed (i.e., transmitted at the same time over the same frequency band);
2) The bandwidth (Hz) of the frequency band utilized for
data transmission; 3) The spectral efficiency (b/s/Hz) of the
transmission to/from the individual devices, which can be
obtained using one or multiple data streams/layers per device.
Suppose we can reach a peak spectral efficiency of 60
b/s/Hz and a maximum bandwidth of 100 GHz, as listed in
Table 2. If these KPIs are achievable simultaneously by the
6G system, then we reach 6 Tb/s. As indicated earlier, it is
unlikely that the KPIs can reach their peak values simultaneously. However, with 100 GHz of spectrum, it is sufficient
with 10 b/s/Hz to reach 1 Tb/s. This spectral efficiency
requirement is rather easy to achieve; two parallel streams
using 16-QAM is sufficient.
Is the 1 Tb/s unreachable when operating in a conventional
millimeter-wave band with “only” 1 GHz of spectrum and
with a spectral efficiency of 10 b/s/Hz? No, this is when the
multiplexing gain is the design variable that can be pushed
to its limit instead. By spatial multiplexing of 100 devices,
the accumulate capacity is 1 Tb/s, even if each device only
achieves 10 Gb/s. It is even possible to reach 1 Tb/s in sub-6
GHz bands by an even more aggressive spatial multiplexing.
These widely different ways of scoring the terabit/s
goal call for a variety of different technical enablers, at
the spectrum level, infrastructure level, as well as protocol/algorithmic level.


**III. ENABLERS AT THE SPECTRUM LEVEL**

One of the three key factors in (1) is the bandwidth. Reaching
1 Tb/s over a point-to-point link will require not only enhanced utilization of current wireless spectrum, but also the
adoption of additional spectrum bands for communications.
Currently, 5G defines operations separately for sub-6 GHz
and 24.25 to 52.6 GHz. Release 17 is expected to extend
the upper limit to 71 GHz [28] and band options up to
114.25 GHz were included in the Release 16 pre-study on
NR beyond 52.6 GHz. In the 6G era, we expect to see an
expansion of the spectrum into many new bands including
additional mmWave bands, the THz band (0.1-10 THz) and
the optical wireless spectrum (including infrared and visible
optical). The potential spectrum regions are illustrated in
Fig. 4. The communication bandwidth is expected to increase at higher frequencies. For example, up to 18 GHz
aggregated bandwidth is available for fixed communications



VOLUME 4, 2021 9


in Europe in the frequency band 71-100 GHz, while in the
USA, both mobile and fixed communications are allowed.
Beyond mmWave, there are also tens of GHz wide bands
between 95 GHz and 3 THz recently opened by the FCC for
experimental use to support the development of innovative
communication systems [29].
When moving to higher frequencies in 6G, the intention
is not to achieve a gradual increase in operational frequency,
as was done in 5G. Instead, we envision a convergence of
existing technologies in these different bands into a joint
wireless interface that enables seamless handover between

bands. The operation in existing bands will be enhanced in
6G with respect to the KPIs described earlier, but not all
targets are expected to be reached in all frequency bands.
For example, low frequency bands are often preferable in
terms of spectral efficiency, reliability, mobility support, and
connectivity density. In contrast, high frequency bands are
often preferable in terms of peak data rates and latency. It
is not a question of one or the other band, but a dynamic
utilization of all bands. When a UE can access several bands,

the network can allocate it to the ones that are most suited for

its current service requests.
There is plenty of experience in operating wireless communication systems in the sub-6 GHz spectrum. In 5G,
moving from sub-6 GHz to mmWave has introduced several
technical challenges ranging from initial access to beamforming implementation since fully digital solutions take
time to develop. The development of 5G has led to large
innovations in these respects. Now, for 6G, all these become
even more challenging when going to higher frequencies and,
thus, new solutions are needed. In this section, we describe

the enablers for THz communications and the enablers for

optical wireless communications in detail.


_**A. ENABLERS FOR TERAHERTZ COMMUNICATIONS**_

THz communication [30], [31] is envisioned as a 6G technology able to simultaneously support higher data rates (in
excess of 1 Tbps) due to the larger bandwidth and denser
networks (hundreds to thousands of spectrum sharing users)
due to the shorter ranges.


1) THz technologies
For many years, the lack of compact, energy-efficient device technologies (which are able to generate, modulate,
detect, and demodulate THz signals) has limited the feasibility of utilizing this frequency range for communications.
However, many recent advancements with several different
device technologies are closing the so-called THz gap [32].
In an _electronic approach_, the limits of standard silicon
CMOS technology [33], silicon-germanium BiCMOS technology [34] and III–V semiconductor transistor [35] and
Schottky diode [36] technologies are being pushed to reach
the 1 THz mark. In a _photonics approach_, uni-traveling
carrier photodiodes [37], photo-conductive antennas [38],
optical down-conversion systems [39] and quantum cascade
lasers [40] are being investigated for THz systems.



N. Rajatheva _et al._ : Scoring the Terabit/s Goal: Broadband Connectivity in 6G


More recently, the use of nanomaterials such as graphene
is enabling the development of plasmonic devices [41]. These
devices are intrinsically small, operate efficiently at THz
frequencies, and can support large modulation bandwidths.
Examples of such devices are, graphene-based plasmonic
THz sources [42], [43], modulators [44], [45], antennas [46],

[47] and antenna arrays [48], [49]. Moreover, graphene is just
the first of a new generation of two-dimensional materials,
which can be stacked to create new types of devices that
leverage new physics.
Clearly, the technology readiness level of the different
approaches will determine the timeline for their adoption
in practical communication systems. The majority of THz
technology demonstrators and communication testbeds are
largely based on electronic systems [50]–[55] consisting
of frequency multiplying chains able to generate a (sub)THz carrier signal from a mmWave oscillator, followed
by a frequency mixer that combines the THz carrier with
an information-bearing intermediate frequency. Purely photonic [56] and hybrid electronic/photonic systems [57] have
also been demonstrated. Ultimately, the adoption of one or
other technology will depend on the application too. For
example, THz transceivers for user-equipment will require
large levels of integration, achievable at this stage only by
CMOS technology. In applications related to backhaul connectivity, high-power III-V semiconductor systems are likely
to be adopted instead.


2) THz Wave Propagation and Channel Modeling
In parallel to device technology developments, major efforts
have been devoted to characterize the propagation of THz
waves and to correspondingly develop accurate channel models in different scenarios.

In the case of LoS propagation in free space, the two
main phenomena affecting the propagation of THz signals
are spreading and molecular absorption. The _spreading loss_
accounts for the attenuation due to expansion of the wave
as it propagates through the medium, and it is determined
by the spreading factor and the antenna effective area, which
measures the fraction of the power a receiving antenna can
intercept. As the carrier frequency increases, the wavelength
becomes smaller (sub-millimetric at THz frequencies), which
leads to smaller antennas (when comparing equal-gain antennas) and this results in a lower received power. The _molecular_
_absorption loss_ accounts for the attenuation that a propagating electromagnetic wave suffers because a fraction of its
energy is converted into vibrational kinetic energy in gaseous
molecules. THz waves can induce internal resonances in

molecules, but are not ionizing. In our frequency range of
interest, water vapor is the main absorber.
In Fig. 5, the spreading and absorption losses are illustrated as functions of the frequency for different distances
for standard atmospheric conditions. These results are obtained with the model presented in [58], which combines
tools from radiative transfer theory, electromagnetics and
communication theory, and leverages the contents of the



10 VOLUME 4, 2021


N. Rajatheva _et al._ : Scoring the Terabit/s Goal: Broadband Connectivity in 6G


**FIGURE 4.** Potential spectrum regions for 6G.


150


100



50





0

0.2 0.4 0.6 0.8 1 1.2 1.4

Frequency [THz]


**FIGURE 5.** Spreading loss (dashed lines) and molecular absorption loss (solid lines) for frequencies ranging from 0.1 to 1.1 THz and three different distances (1,
10 and 100 m).



high-resolution transmission molecular absorption database.
Molecular absorption defines multiple transmission windows
with bandwidths ranging from tens of GHz up to hundreds
of GHz depending on the transmission distance and the
molecular composition of the medium. For the time being,
the majority of THz efforts have been focused on the transmission windows at 120-140 GHz [59], [60], 240 GHz [57],

[61], 300 GHz [50], [62], [63] and 650 GHz [51], [53].
Within these transmission windows, the molecular absorption losses are much lower than the spreading loss. The latter
requires the use of high-gain directional antenna systems (or
arrays of low-gain antennas with the same physical aperture),
to compensate for the limited transmission power of THz

sources.


Other meteorological factors, such as rain and snow, cause
extra attenuation on the THz wave propagation. In [64], the
the rain attenuation was calculated at 313 GHz and 355 GHz

by the four raindrop size distributions Marshall-Palmer, Best,
Polyakova-Shifrin, and Weibull raindrop-size distributions,
and a specific attenuation model using the prediction method
recommended by ITU-R P.838-3 [65]. The results show
that the propagation experiment results are in line with the
specific attenuation prediction model recommended by ITUR. In [66], snow events were categorized as dry snow and



wet snow. The comparison between the measured attenuation
through wet and dry snowfall showed that a larger attenuation happened through wet snowfall. The general conclusion
was that as the snowfall rate increases, the attenuation also
increases for both wet and dry snow. Higher attenuation is
expected for snowflakes with higher water content. Yet, till
now, few experimental studies on snow effect on the wireless
channel have been reported systematically compared to the
rain effect, not to mention in the THz band range. Thus,
there is no recommendation in ITU-R to predict the snow
attenuation. In [67], an extra loss of 2.8 dB was measured

for a distance of 8 m in a LoS link at 300 GHz under the
condition of the most significant snowstorm. This implies
that the link length of THz outdoor communications can
differ considerably under various meteorological conditions.


Beyond free-space propagation, the presence of different
types of elements (e.g., objects, furniture, walls, plants, animals, and human beings) affect the propagation of THz
signals in realistic scenarios. Depending on the material,
shape and dimensions, THz signals might be transmitted,
absorbed, reflected or diffracted. For example, THz signals
propagate well across common plastics. In the case of paper,
cloth and wood, THz signals are partially reflected, partially
absorbed and partially transmitted. Metals, glass, and tiles



VOLUME 4, 2021 11


N. Rajatheva _et al._ : Scoring the Terabit/s Goal: Broadband Connectivity in 6G



with different coatings are mainly reflectors. Such reflection
can be of two types—specular or diffused—depending on
the roughness of the surfaces relative to the signal wavelength [68]–[71].
For the time being, a few multi-path channel models for the
THz band have been developed [72]–[75]. In [72], Rayleigh,
Gaussian and negative exponential PDFs were fitted to extensive data points obtained by means of ray-tracing simulations
of an office scenario. In [73], a ray-tracing approach was
followed to characterize the multi-path THz channel again
in an office-like scenario. For these two works, the main
constraint in their approach is the need to re-run extensive
simulations for each possible scenario. In [74], a stochastic
multi-path channel model for an infinite field of scatterers
was analytically derived and semi-closed form expressions
for the frequency autocorrelation function were computed.
In [75], an analytical model for the number of single bounce
multi-path components and the power delay profile in a
rectangular deployment scenario is derived, considering the
density of obstacles, variable geometry of the rectangle, the
signal blocking by obstacles and the propagation properties
of THz signals.
To experimentally characterize the THz channel, channel
sounding techniques are needed. However, channel sounding
at the THz band is more challenging than that at traditional
cellular bands, mainly because the directionality and attenuation of THz signals are much greater than those at frequencies
below 6 GHz [76]. In order to overcome high path loss and
capture energy from all the directions, channel sounders for
high frequencies (such as millimeter wave or THz) often
resort to manually or mechanically rotate high-gain steerable horn antennas [77], [78]. In such measurements, only
a relatively small number of channel samples with limited
degrees of freedom (e.g., two-dimensional channel) can be
obtained, because the measurements are very costly and timeconsuming. Thus, the relatively small measurement data sets
must be extended using simulation-based analysis to extract
spatial and temporal channel parameters, like the authors
of [79], [80] did for urban cellular channels at mmWave
bands. Thus, employing the ray-tracing simulators which
are calibrated by measurements becomes an alternative to
extending the sparse empirical data sets and analyzing the
three-dimensional channel characteristics at mmWave bands

[79], [81], [82].
As an example in this direction, an _M_ -sequence (a kind
of pseudo random sequence) correlation based THz UWB
channel sounder is customized [84]. In order to extend the
limited channel measurements to more general cases, an
in-house-developed HPC cloud-based 3D RT simulator –
CloudRT – is jointly being developed by Beijing Jiaotong
University and Technische Universität Braunschweig, integrating a V2V RT simulator [85], [86] and a UWB THz RT
simulator [87]. In our recent work [88], [89], CloudRT has so

far been validated at 30 GHz and 90 GHz in HST outdoor and

tunnel environments, respectively. Based on image theory,
CloudRT can output more than 10 properties of every ray,















**FIGURE 6.** Measurement campaign in a real HST wagon,
originally from Fig. 1 of [83].



-90


-95


-100


-105


-110


-115


-120


-125


-130


-135


-140





0 50 100 150 200 250 300 350

Delay [ns]


**FIGURE 7.** Comparison of PDP between measurement
and RT simulation at 300 GHz, originally from Fig. 12
of [83].


including the type of the ray, reflection order, time of arrival,
complex amplitude, AoA, AoD, EoA, EoD, and so on. More
information on CloudRT can be found in tutorial [90] and
[http://www.raytracer.cloud.](http://www.raytracer.cloud)
For THz channel characterization, the THz UWB channel
sounder and the CloudRT have been utilized in multiple
scenarios such as smart rail mobility. Fig. 6 shows the measurement campaign in a real HST wagon between 300-308
GHz with an 8 GHz bandwidth. The comparison of PDP
between the measured and simulated PDP of the intra-wagon
scenario is shown in Fig. 7, where the good agreements of
the significant paths are achieved both in terms of power
and delay. This measurement-validated RT simulator can be
utilized to generate more realistic channel data with various
setups for comprehensive characterization. More information
can be found in [83].
While more extensive multi-path propagation modeling
studies are needed, some of the key properties of the THz
channel are already clear. While multiple paths are expected,
the combination of high-gain directional antennas at the



12 VOLUME 4, 2021


N. Rajatheva _et al._ : Scoring the Terabit/s Goal: Broadband Connectivity in 6G


transmitter and the receiver to overcome the otherwise very
large propagation losses and the fact that many materials
behave as THz (partial) absorbers limits the number of signal
copies at the receiver. It is likely that the strongest copies
of the signal are generated in the vicinity of the receiver,
presumably by metallic objects. It is worth noting that, as
a result of the very small wavelength of THz signals, very
small metallic obstacles (such as a paper clip) can significantly reflect or diffract THz signals (a phenomenon that can
be leveraged for eavesdropping [91]). Ultimately, the THz
channel is extremely scenario-dependent, which makes the
development of real-time ultra-broadband channel estimation
and equalization a key task at the physical layer.


3) Signal Processing, Communications and Networking
Challenges
The capabilities of the THz devices and the peculiarities of
the THz channel introduce opportunities and challenges for
THz communication networks.

On the one hand, the much higher spreading losses combined with the low power of THz transmitters makes increasing the communication distance _the biggest challenge_ [92].
Despite their quasi-optical propagation traits, THz communications possess several microwave characteristics. As a
result, the enablers at the infrastructure and algorithm levels
introduced in Sec. IV and Sec. V are highly relevant for
THz communications. For example, the term ultra-massive
MIMO was first introduced precisely in the context of THz
communications [93]. Since then, many potential architectures have been proposed, ranging from adaptive arrays-ofsubarrays antenna architectures in which each subarray undergoes independent analog beamforming [94], [95], to fully
digital beamforming architectures with thousands of parallel
channels [49] enabled by the aforementioned graphene-based
plasmonic devices. Similarly, IRSs at THz frequencies have
also been proposed to overcome obstacles through directed
NLoS propagation as well as for path diversity [96]–[98].
On the other hand, at higher frequencies, molecular absorption has a higher impact. As already explained, the
absorption defines multiple transmission windows, tens to
hundreds of GHz wide each. As a result, simple singlecarrier modulations can already enable very high-speed
transmissions, exceeding tens of Gbps. Beyond the traditional schemes, dynamic-bandwidth algorithms that can cope
with the distance-dependent absorption-defined transmission
channel bandwidth have been proposed for short [99] and
long [100] communication distances. Ultimately, resource
allocation strategies to jointly orchestrate frequency, bandwidth and antenna resources need to be developed.
An additional challenge in making use of the most of the
THz band is related to the digitalization of large-bandwidth
signals. While the THz-band channel supports bandwidth
in excess of 100 GHz, the sampling frequency of stateof-the-art digital-to-analog and analog-to-digital converters
is in the order of 100 Gigasamples-per-second. Therefore,
high-parallelized systems and efficient signal processing are



needed to make the most out of the THz band. Since channel

coding is the most computationally demanding component
of the baseband chain, efficient coding schemes need to be
developed for Tbps operations. Nevertheless, the complete
chain should be efficient and parallelizable. Therefore, algorithm and architecture co-optimization of channel estimation,
channel coding, and data detection is required. The baseband
complexity can further be reduced by using low-resolution
digital-to-analog conversion systems, and all-analog solutions should also be considered.

Beyond the physical layer, new link and network layer
strategies for ultra-directional THz links are needed. Indeed,
the necessity for very highly directional antennas (or antenna
arrays) simultaneously at the transmitter and at the receiver
to close a link introduces many challenges and requires a
revision of common channel access strategies, cell and user
discovery, and even relaying and collaborative networks. For
example, receiver-initiated channel access policies based on
polling from the receiver, as opposed to transmitter-led channel contention, have been recently proposed [101]. Similarly,
innovative strategies that leverage the full antenna radiation
pattern to expedite the neighbor discovery process have been
experimentally demonstrated [102]. All these aspects become
more challenging for some of the specific use cases defined in
Section II, for example, in the case of wireless backhaul, for
which very long distances lead to very high gain directional
antennas and, thus, ultra-narrow beamwidths, or smart rail
mobility, where ultra-fast data-transfers can aid the intermittent connectivity in train-to-infrastructure scenarios.
Last but not least, it is relevant to note that there is
already an active THz communication standardization group,
IEEE 802.15 IGTHz, which lead to the first standard IEEE
802.15.3d-2017.


_**B. ENABLERS FOR OPTICAL WIRELESS**_

_**COMMUNICATIONS**_
OWC is an efficient and mature technology that has been
developed alongside the cellular technology, which has only
used radio spectrum. OWC can potentially satisfy the demanding requirements at the backhaul and access network
levels in beyond 5G networks. As the 6G development gains
momentum, comprehensive research activities are being carried out on the development of OWC-based solutions capable
of delivering ubiquitous, ultra-high-speed, low-power consumption, highly secure, and low-cost wireless access in diverse application scenarios [103]. In particular, this includes
the use of hybrid networks that combine OWC with radio
frequency or wired/fiber-based technologies. Solutions for
IoT connectivity in smart environments is being investigated
for developing flexible and efficient backhaul/fronthaul OWC
links with low latency and support for access traffic growth

[104].
The OWC technology covers the three optical bands of infrared (IR: 187-400 THz, 750-1600 nm wavelength), visible
light (VL: 400-770 THz, 390-750 nm) and ultraviolet (UV:
1000-1500 THz, 200-280 nm). FSO and VLC are commonly



VOLUME 4, 2021 13


used terms to describe various forms of OWC technology

[105]. FSO mainly refers to the use of long-range, high speed
point-to-point outdoor/space laser links in the IR band [106],
while VLC relies on the use of LEDs operating in the VL
band, mostly in indoor and vehicular environments [107].
In comparison to RF, OWC systems offer significant technical and operational advantages including, but not limited
to: i) huge bandwidth, which leads to high data rates; e.g.,
a recent FSO system achieved a world record data rate of
13.16 Tbps over a distance of 10 km [105], and multiple
Gbps in indoor VLC setups [108]; However, it is difficult
to make fair comparisons in terms of how many bits/Joule
that are delivered by VLC systems. This is mainly due to the
fact that VLC is also employed for illumination purpose. In
this respect, if the power spent for illumination is considered
as a completely wasted power, the power efficiency value of
VLC would be around of, or lower than, the 1-10 Gb/Joule
value reached by RF in 5G. On the other hand, it is possible
to reach high bit rate values per Joule by using small energies
in VLC’s imaging and coherent applications. ii) operation
in unregulated spectrum, thus no license fees and associated
costs; iii) immunity to the RF electromagnetic interference;
iv) a high degree of spatial confinement, offering virtually
unlimited frequency reuse capability, inherent security at the
physical layer, and no interference with other devices; v)
a green technology with high energy efficiency due to low
power consumption and reduced interference. With such features, OWC is well positioned to be a prevailing complement
to RF wireless solutions from micro- to macro-scale applications, including intra/inter-chip connections, indoor WA
and localization, ITS, underwater, outdoor and space pointto-point links, etc. Beyond the state-of-the-art, however, the
dominance of RF-based WA technologies will be challenged
and, in order to release the pressure on the RF spectrum,
utilization of the optical transmission bands will be explored
to address the need of future wireless networks beyond 5G.
However, some disadvantages of OWC, compared to RF,
needs to be addressed here for completeness. These are:


_•_ Tx-Rx alignment/collimation requirements (for longrange free-space OWC; this can be a disadvantage
compared to sub-6 GHz RF systems but this is also a
requirement for mmWave RF systems).

_•_ Limitations on receive sensitivity due to the existing
detector technology (it is not an issue in RF systems).


Optimized solutions need to be devised for the integration
of OWC in heterogeneous wireless networks as well as for
enhanced cognitive hybrid links in order to improve reliability in coexisting optical and RF links. Moreover, more
efforts are needed in characterizing the propagation channel
considering the additional practical constraints imposed by
the optical front-ends, as well as developing optimal physical
layer design and custom-designed MAC and NET layers. In
addition, standardization efforts need to be undertaken in
the area of OWC to standardization bodies such as IEEE,
ITU, ETSI, 3GPP and international forums such as WWRF,



N. Rajatheva _et al._ : Scoring the Terabit/s Goal: Broadband Connectivity in 6G


**FIGURE 8.** An indoor VLC system configuration.


Photonics21 and 5G-PPP before it is utilized in 6G.

VLC, a special form of OWC, is also called LiFi [108] in
the same way that WLAN systems are today widely known
as WiFi. LiFi is a promising technology to provide local
broadband connectivity [107]. As shown in Fig. 8, VLC
provides high-speed, bi-directional, networked delivery of
data through the lighting infrastructure. When a device moves
out of the light cone of one light source, the services can
be handed over to the next light source, or eventually, the
device can be connected and handed over to an RF-based

system, if optical access is not any longer provided. The
former case corresponds to a horizontal handover while the
latter is known as vertical handover, and these handovers
are needed to provide seamless connectivity to the end-user
device. In VLC, all the baseband processing at the transmitter
and the receiver is performed in the electrical domain and
intensity modulation/direct detection is the most practical
scheme. LEDs with a large FoV or laser diodes with small
FoV are used to encode and transmit data over the LoS/NLoS

optical channel. Photodetectors at the receiver convert datacarrying light intensity back to electrical signals for baseband
processing. However, the intensity-modulating data signal
must satisfy a positive-valued amplitude constraint. Hence, it
is not possible to straightforwardly apply techniques used in
RF communications. A VLC-enabled device inside a pocket
or briefcase cannot be connected optically, which is one
example of why a hybrid optical-radio wireless network is
needed. A reconfigurable optical-radio network is a highperformance and highly flexible communications system that
can be adapted for changing situations and different scenarios

[109], [110].
Performance-wise, data throughput below 100 Mbps can
be achieved with relatively simple optical transceivers and
off-the-shelf components. Data rates of up to hundreds of
Gbps have been demonstrated in laboratory conditions, and it
is expected that even Tbps-communications will be achieved
in the future. Key optical components for VLC, such as LEDs
and photodetectors, have been developed for decades, and
they are typically standard low-cost components. VLC is not
intended to replace but complement existing technologies.
When it comes to vertical applications, VLC can be used for



14 VOLUME 4, 2021


N. Rajatheva _et al._ : Scoring the Terabit/s Goal: Broadband Connectivity in 6G


both conventional data services for consumers and support
emerging applications and services such as smart cities, smart
buildings, e-health, factories of the future, ITS, smart grids,
mining, and the IoT. The concept of LIoT exploits light not
only to create optical links but also to harvest its energy

[111], [112]. Thus, an LIoT node can be energy autonomous.


Open research directions in OWC as well as in VLC
toward 6G include:


_•_ Accurate VLC channel modeling and characterization
for various deployment scenarios with a particular emphasis on user-dense environments. Incorporating user
mobility and device orientation into the VLC channel
models and combining VLC and RF systems [113]–

[115].

_•_ New non-coherent physical-layer transmission schemes
such as spatial modulation, and its variations can
be used, as well as non-orthogonal communication
schemes such as MIMO [116]–[121].

_•_ Exploiting red-green-blue LEDs, development of new
materials and optoelectronic devices (e.g., fast nonphosphorous LEDs, micro-LEDs), very fast switching
mechanisms between optical and radio systems, etc.

[122].

_•_ Use of OWC to provide secure and safe connectivity in in-body communications applications, including
communications to and from the body, communications
between sensors inside the body, etc. Recent results
have shown that near infrared light can be used for this
purpose [123], [124].

_•_ Design of new and novel optical IoT, new devices and
interfaces to tackle the efficient generation, routing,
detection and processing of optical signals [125].

_•_ For ultra-dense IoT scenarios, there are a number of
open challenges that appeal for radical rethinking of
network topologies and the design of media access
control and network layers in OWC [126].

_•_ In VLC, to account for multi-user scenarios and user
mobility, robust low-complexity multiple access techniques need to be designed together with efficient cellular architectures with user-scheduling and intra-room
handover capability, achieving high capacity, low latency, and fairness [104], [127].

_•_ At the MAC layer, due to the small packet sizes used
in M2M applications and constraints on sensor devices,
robust link quality estimators will be developed and
routing algorithms will be devised taking into account
the optimal tradeoff between the link capacity, connectivity, latency and energy consumption [107], [128],

[129].

_•_ In medium-range OWC, the effects of weather and environmental conditions, ambient noise, and link misalignments need to investigated and to enable connectivity
between distant vehicles, physical-layer designs need
to be built upon multi-hop transmission to reduce the
delay, which is very important in the transmission of



road safety related information [104], [107], [129].

_•_ For long-range links, extensive research should be carried out for the minimization of the terminal size to

enable the technology to be integrated in small satellites,
e.g., CubeSats, with data rates up to 10 Gbps and to investigate how to deal with cloud obstruction, site diversity techniques and smart route selection algorithms will
be investigated for satellite links and airborne networks,
respectively. Also, hybrid RF/FSO and optimized multihop transmission techniques will also be investigated
to improve link reliability between satellites or HAPS

[130]–[132].


**IV. ENABLERS AT THE INFRASTRUCTURE LEVEL**

Another key factor in the capacity formula in (1) is the
multiplexing gain. There is a variety of ways that one can
increase the number of simultaneously active UEs in 6G
systems, including using larger antenna arrays and distributed
arrays. This section will cover these technologies, as well as
other infrastructure-related 6G enabling technology.


_**A. ULTRA-MASSIVE MIMO AND HOLOGRAPHIC RADIO**_

Massive MIMO is a cellular technology where the APs are
equipped with a large number of antennas, which are utilized
for spatial multiplexing of many data streams per cell to one
or, preferably, multiple users. Massive MIMO has become
synonymous with 5G, but the hardware implementation and
algorithms that are used in practice differ to a large extent
from what was originally proposed in [133] and then described in textbooks on the topic [134], [135]. For example,
compact 64-antenna rectangular panels with limited angular
resolution in the azimuth and elevation domains are adopted
instead of physically large horizontal uniform linear arrays
with hundreds of antennas [13], which would produce very
narrow azimuth beams. Moreover, a beam-space approach is
taken by dividing the spatial domain into grids of 64 beams
pointing to predetermined angular directions using twodimensional DFT codebooks. Only one of these predefined
beams is selected for each user, thus the approach is only
appropriate for LoS communications with calibrated planar
arrays and widely spaced users. In general, NLoS channels
contain arbitrary linear combinations of these beams, the arrays might have different geometries, and the array responses
of imperfectly calibrated arrays cannot be described by DFT
codebooks. A practical reason for these design simplifications is that analog and hybrid beamforming were needed to
come quickly to market in 5G. However, fully digital arrays
will be available for a wide range of frequencies (including
mmWave bands) when 6G arrives and, therefore, we should
utilize them to implement something capable of approaching
the theoretical performance of massive MIMO [134], [135].
Since the massive MIMO terminology has become diluted by
many suboptimal design choices in 5G, we will use the term
_ultra-massive MIMO_ [93] in this paper to describe a potential
6G version of the technology.



VOLUME 4, 2021 15


1) Beamforming Beyond the Beam-Space Paradigm
Spectrum resources are scarce, particularly in the sub-6
GHz bands that will always define the baseline coverage
of a network. Hence, achieving full utilization of the spatial dimensions (i.e., how to divide the available spatial
resources between concurrent transmissions) is particularly
important. The current beam-space approach describes the
signal propagation in three dimensions and, although current
planar arrays are capable of generating a set of beams with
varying azimuth and elevation angles, it remains far from
utilizing all the available spatial dimensions. On the other
hand, a 64-antenna array is capable of creating beams in a
64-dimensional vector space (where most beams lack a clear
angular directivity but can anyway match a physical channel)
and utilize the multipath environment to focus the signal at
certain points in space.
The low-dimensional approximation provided by the twodimensional DFT-based beam-space approach can be made
without loss of optimality only in a propagation environment
with no near-field scattering, predefined array geometries,
perfectly calibrated arrays, and no mutual coupling. These
restrictions are impractical when considering ultra-massive
MIMO arrays that are likely to interact with devices and
scattering objects in the near-field, have arbitrary geometries,
and feature imperfect hardware calibration and coupling effects. In particular, the latter stems from the non-zero mutual
reactance between (hypothetical) isotropic radiators whose
spacing is a multiple of half a wavelength [136]. That is
why the massive MIMO concept was originally designed to
use uplink pilots and uplink-downlink channel reciprocity to
estimate the entire channel instead of its three-dimensional
far-field approximation [134], [135]. Mutual coupling at the
transmit side affects not only the overall channel but also the
amount of radiated power, which in turn impacts the reciprocity of the so-called information theoretic channel [137].
The effects of mutual coupling can be taken into account
using models based on circuit theory [138]–[140].
The beamforming challenge for 6G is to make use of physically large panels, since the dimensionality of the beamforming is equal to the number of antennas and the beamwidth is
inversely proportional to the array aperture. With an ultrahigh spatial resolution, each transmitted signal can be focused on a small region around the receiver, leading to a
beamforming gain proportional to the number of antennas
as well as enhanced spatial multiplexing capabilities. The
latter is particularly important to make efficient use of the low
frequency bands where the channel coherence time is large
and thereby accommodate the channel estimation overhead
for many users. With a sufficient number of antennas, 1 MHz
of spectrum in the 1 GHz band can give the same data rate
as 100 MHz of spectrum in the 100 GHz band. The reason
for this is that the lower frequency band supports spatial
multiplexing of 100 times more users since the coherence
time is 100 times larger. Recently, novel solutions to reduce
the channel estimation overhead and support more users in
higher frequency bands have been proposed [141], [142].



N. Rajatheva _et al._ : Scoring the Terabit/s Goal: Broadband Connectivity in 6G


Ideally, ultra-massive MIMO should be implemented using fully digital arrays with hundreds or thousands of phasesynchronized antennas. This is practically possible in both
sub-6 GHz and mmWave bands [13], although the implementation complexity grows with the carrier frequency. On
the one hand, new implementation concepts are needed that
are not stuck in the suboptimal beam-space paradigm, as in
the case of hybrid beamforming, but can make use of all the
spatial dimensions. On the other hand, new device technologies, possibly leveraging new materials, can be utilized to
implement on-chip compact ultra-massive antenna arrays that
can potentially enable fully digital architectures [49]. Doubly
massive MIMO links, wherein a large number of antennas
are present at both sides of the communication link, will also
be very common at mmWave frequencies [143]–[146]. Note
that orbital angular momentum methods cannot be used to
increase the channel dimensionality of MIMO links since
these dimensions are already implicitly utilized by the MIMO
technology [147].
Continuous-aperture antennas can be considered to improve the beamforming accuracy. While the beamwidth of
the main-lobe is determined by the array size, a continuous aperture gives cleaner transmissions with smaller sidelobes [148]. One possible method to implement such technology is to use a dense array of conventional discretely
spaced antennas, but the cost and energy consumption would
be prohibitive if every antenna element has an individual
RF chain. Another method is to integrate a large number
of antenna elements into a compact space in the form of a
meta-surface, but this would be limited to the passive setup
described in Section IV-B. A possible solution to implement
active continuous-aperture antennas is given by the holographic radio technology described next.


2) Holographic Radio
Holographic radio is a new method to create a spatially
continuous electromagnetic aperture to enable holographic
imaging-level, ultra-high density spatial multiplexing with
pixelated ultra-high resolution [149]. In general, holography records the electromagnetic field in space based on the
interference principle of electromagnetic waves. The target
electromagnetic field is reconstructed by the information
recorded by the interference of reference and signal waves.
The core of holography is that the reference wave must be
strictly coherent as a reference and the holographic recording
sensor must be able to record the continuous wave-front

phase of the signal wave so as to record the holographic electromagnetic field with high accuracy. Because radio and light
waves are both electromagnetic waves, holographic radio
is very similar to optical holography [13]. For holographic
radios, the usual holographic recording sensor is the antenna.


**Realization of holographic radio**
To achieve a continuous-aperture antenna array, one ingenious method is to use an ultra-broadband tightly coupled
antenna array based on a current sheet. In this approach, uni


16 VOLUME 4, 2021


N. Rajatheva _et al._ : Scoring the Terabit/s Goal: Broadband Connectivity in 6G


traveling-carrier photodetectors are bonded to the antenna
elements using flip-chip technology and form a coupling
between antenna elements [150]. In addition, patch elements
are directly integrated into the electro-optic modulator. The
current output by the photodetectors directly drives the antenna elements, so the entire active antenna array has a very
large bandwidth (about 40 GHz). Moreover, this innovative
continuous-aperture active antenna array does not require an
ultra-dense RF feed network at all, which means not only
that it is achievable but also that is has clear implementation
advantages.
Unlike the beam-space approach to massive MIMO, which
dominates in 5G, holographic radios are capable of making
use of all available spatial dimensions. The signal wavefront
is generated by exploiting a diffraction model based on the
Huygens-Fresnel principle. Each point on the array is emitting a spherical wave and the interference pattern between
these waves is creating the emitted waveform. The set of
possible waves that can be generated is called the holographic
radio space and is illustrated in Fig. IV-A2. Correspondingly,
accurate computation of the communication performance
requires detailed electromagnetic numerical computations
for the radio space; that is, algorithms and tools related to
computational electromagnetics and computational holography. The spatial channel correlation is described based on
the Fresnel-Kirchhoff integral. Moreover, holographic radios
use holographic interference imaging to obtain the RF spectral hologram of the RF transmitting sources (e.g., UEs).
In other words, the CSI is implicitly acquired by sending
pilot signals from the UE and apply holographic image
recording techniques to find the interference pattern that
maximizes the received signal power. A three-dimensional
constellation of distributed UEs in the RF phase space can
be obtained through spatial spectral holography, providing
precise feedback for spatial RF wave field synthesis and
modulation in the downlink. Spatial RF wave field synthesis
and modulation can obtain a three-dimensional pixel-level
structured electromagnetic field (similar to an amorphous
or periodic electromagnetic field lattice), which is the highdensity multiplexing space of holographic radio and different
from sparse beam-space considered in 5G.
It is worth noting that all waveforms that can be generated
using a holographic array can in theory be also generated by
replacing the holographic array with a sphere (that would
have encapsulated it) whose surface area consists of halfwavelength-spaced discrete antennas. We can achieve the
same communication performance using such a spherical
array, along with the classical theory for channel estimation
and signal processing in the massive MIMO literature [135].
However, this is a hypothetical comparison and not a practical alternative. The purpose of holographic radios is to enable
array capabilities that are not practical to implement using
other hardware technologies. On the other hand, since holographic radio utilizes the advantages of optical processing,
spectrum computing, large-scale photon integration, electrooptical mixing, and analog-digital photon hybrid integration



technologies, the physical layer technology must be adapted
to make use of these new methods.


**Signal processing for holographic radio**
There are different ways to implement holographic radios
for the purpose of joint imaging, positioning, and wireless communications [151]. However, extreme broadband
spectrum and holographic RF generation and sensing will
produce massive amounts of data, which are challenging to
process for critical tasks with low latency and high reliability.
Thus, machine learning might be required to operate the
system effectively and bridge the gap between theoretical
models and practical hardware effects. To meet the 6G challenges of energy efficiency, latency, and flexibility, a hierarchical heterogeneous optoelectronic computing and signal
processing architecture will be an inevitable choice [152].
Fortunately, holographic radios achieve ultra-high coherence
and high parallelism of signals through coherent optical upconversion of the microwave photonic antenna array, which
also facilitate the signal processing directly in the optical
domain. However, it is challenging to adapt the signal processing algorithms of the physical layer to fit the optical
domain.

How to realize holographic radio systems is a wide-open
area. Due to the lack of existing models, in future work,
holographic radio will need fully featured theory and modeling converging the communication and electromagnetic theories. Moreover, performance estimation of communication
requires dedicated electromagnetic numerical computation,
such as the algorithms and tools related to computational
electromagnetics and computational holography. The massive MIMO theory can be extended to make optimal use of
these propagation models. As mentioned above, a hierarchical and heterogeneous optoelectronic computing architecture
is a key to holographic radio. Research challenges related to
the design of the hardware and physical layer include the
mapping from RF holography to optical holography, integration between photonics-based continuous-aperture active
antennas, and high-performance optical computing. [1]


_**B. INTELLIGENT REFLECTING SURFACES**_

When the carrier frequency is increased, the wireless propagation conditions become more challenging due to the larger
penetration losses and lower level of scattering, leading
to fewer useful propagation paths between the transmitter
and the receiver. Moreover, designing coherently operating
antenna arrays becomes more difficult since the size of
each antenna element shrinks with the wavelength. In such
situations, an IRS can be deployed and utilized to: _i)_ improve the propagation conditions by introducing additional
scattering, and _ii)_ control the scattering characteristics to
create passive beamforming towards the desired receivers to
achieve high beamforming gain and suppress the co-channel
interference [153], [154]. Ideally, an IRS would create a


1 The authors would like to thank Danping He ( _Beijing Jiaotong Univer-_
_sity_ ) for the fruitful discussions on the topics of this section.



VOLUME 4, 2021 17


N. Rajatheva _et al._ : Scoring the Terabit/s Goal: Broadband Connectivity in 6G



smart, programmable, and controllable wireless propagation
environment, which brings new degrees of freedom to the optimization of wireless networks (in addition to the traditional
transceiver design).
An IRS is a type of relay with innovative hardware features [155]. Instead of being a compact unit, it consists of
a thin two-dimensional surface with a large area. Currently,
the most promising way of implementing an IRS is by
means of a meta-surface consisting of meta-materials with
unusual electromagnetic properties that can be controlled
without the need for traditional RF chains. A large IRS can
be potentially produced at very low cost, complexity, and
energy consumption since no RF components are required
unlike conventional active MIMO arrays and holographic
radio [154], [156], [157]. From an implementation standpoint, IRSs can be conveniently coated on facades of outdoor
buildings or indoor walls/ceilings, making them deployable
with low complexity and potentially invisible to the human
eye. An IRS can be flexibly fabricated and to be mounted
on arbitrarily shaped surfaces and, therefore, to be straightforwardly integrated into different application scenarios. The
integration of an IRS into a wireless network can be made
transparent to the users, thus providing high deployment
flexibility [154].
The IRS concept has its origin in reflectarray antennas,
which are a class of directive antennas that are flat but
can be configured to act as parabolic reflectors or convex
RF mirror [158], [159]. A characteristic feature of an IRS

is that it is neither co-located with the transmitter nor the
receiver, thus making it a relay rather than a transceiver. [2]

The IRS is envisaged to be reconfigurable in real-time so it
can be adapted to small-scale fading variations (e.g., due to


2 An IRS that is co-located with the transmitter or receiver falls more into
the category of holographic radio.



mobility of the users). An IRS contains a large number of
sub-wavelength-sized elements with controllable properties
(e.g., impedance) that can be tuned to determine how an
incoming signal is scattered (e.g., in terms of phase delay,
amplitude, and polarization). As a consequence, an incoming
waveform can be reflected in the shape of a beam whose
direction is determined by the phase-delay pattern over the
elements and can, thus, be controlled [160]. In order words,
the individually scattered signals will interfere constructively
in the desired directions and destructively in other directions.

To exemplify the basic operation, let us consider a system
with a single-antenna transmitter, a single-antenna receiver,
and an IRS with _N_ elements. The direct path is represented
by the channel coefficient _[√]_ _β_ d _e_ _[jψ]_ [d], where _β_ d _>_ 0 is the
channel gain and _ψ_ d _∈_ [0 _,_ 2 _π_ ) is the phase-delay. There are
also _N_ scattered paths, each travelling via one of the IRS
elements. The _n_ th such path is represented by the channel
coefficient _[√]_ _β_ IRS _e_ _[j]_ [(] _[ψ]_ _n_ [TX] + _ψ_ _n_ [RX] _−φ_ [IRS] _n_ ), where _β_ IRS _>_ 0 is the
end-to-end channel gain (i.e., the product of the channel gain
from the transmitter to the IRS and the channel gain from
the IRS to the receiver), _ψ_ _n_ [TX] _∈_ [0 _,_ 2 _π_ ) is the phase-delay
between the transmitter and the IRS, and _ψ_ _n_ [RX] _∈_ [0 _,_ 2 _π_ )
is the phase-delay between the IRS and the receiver. These
parameters are determined by the propagation environment
and thus uncontrollable. However, each element in the IRS
can control the time delay that is incurred to the incident
signal before it is re-radiated, leading to a controllable phasedelay _φ_ [IRS] _n_ _∈_ [0 _,_ 2 _π_ ] for each element _n_ . The received signal
is



(2)







+



� _β_ IRS _e_ _[j]_ [(] _[ψ]_ _n_ [TX] + _ψ_ _n_ [RX] _−φ_ [IRS] _n_ )

~~�~~ � ~~�~~ �
Scattered path from _n_ th IRS element







 _x_ + _w,_




_N_
�


_n_ =1



�



_y_ =



 ~~�~~

�





~~�~~ _β_ d _e_ _[jψ]_ [d]

� ~~��~~ ~~�~~
Direct path



18 VOLUME 4, 2021


N. Rajatheva _et al._ : Scoring the Terabit/s Goal: Broadband Connectivity in 6G


9


8



7


6


5


4


3


2


1





0
0 200 400 600 800 1000


**FIGURE 10.** By adding an IRS to a baseline communication system, the
spectral efficiency can be greatly improved. The gain is particularly large when
the direct path is weak, as in Case 1.


where _x_ is a desired signal with power _P_ and _w ∼CN_ (0 _, σ_ [2] )
is the receiver noise. The signal-to-noise ratio (SNR) in (2) is



2



SNR = _[P]_

_σ_ [2]



�����



~~�~~ _β_ d _e_ _[jψ]_ [d] +



~~�~~



_N_
�


_n_ =1



�



_β_ IRS _e_ _[j]_ [(] _[ψ]_ _n_ [TX] + _ψ_ _n_ [RX] _−φ_ [IRS] _n_ )
�����



and are sub-wavelength-sized, thus more than 1000 elements
fits into a 1 m _×_ 1 m surface.
When the IRS elements are configured to maximize the
SNR, all the propagation paths will interfere constructively at
the location of the receiver. If the receiver is in the vicinity of
the IRS, the reflected signal will be focused on the particular
spatial location of the receiver. As this location is moved further away, the focusing operation gradually becomes equivalent to forming a beam in the angular direction leading to
the point, as illustrated in Fig. 11. Since an IRS can change
not only the direction of the reflected wave but also the shape
of the waveform, it should be viewed as an electronically reconfigurable curved/concave mirror. There are special cases
when an IRS can approximate a specular reflector (i.e., a flat
mirror) but this is generally suboptimal [155], [162].
The IRS is a full-duplex transparent relay that does not
amplify the incident signal but reflects it without causing any
noticeable propagation delays (except that the delay spread
of the channel might increase) [163], [164]. Since the surface
needs to be physically large to beat a classical half-duplex
regenerative relay and is subject to beam-squinting [162],

[165], the most promising use case for IRS is to increase
the propagation conditions in short-range communications,
particularly in sub-THz and THz frequency bands where conventional relaying technology is unavailable. For example, an
IRS can provide an additional strong propagation path that
remains available even when the LoS path is blocked. An IRS
should ideally be deployed to have an LoS path to either the
transmitter and/or receiver. There are also possible use cases
in NLOS scenarios, for example, to increase the rank of the
channel to achieve the full multiplexing gain.
To further elaborate on how to deploy an IRS, we continue
the example from Fig. 10 and focus on Case 2, where the
direct LOS path between the transmitter and receiver has a
channel gain of _−_ 75 dB. The distance between the transmitter and receiver is 45 m. An IRS with _N_ = 1024 elements

is deployed somewhere along a line that is 5 m from the
line between the transmitter and the receiver. We compute
the channel gains of the reflected LoS paths using the formulas from [162]. Fig. 12 shows the end-to-end channel gain
_|_ _[√]_ _β_ d + _N_ _[√]_ _β_ IRS _|_ [2] achieved for different locations of the IRS.
The baseline channel gain _β_ d = _−_ 75 dB is also shown as a
reference. We notice that the IRS improves the channel gain
the most when it is either close to the transmitter or close

to the receiver, but there is always a substantial gap over the
baseline case with no IRS. Hence, two preferable types of
IRS deployments are envisioned:

_i)_ A large IRS can be deployed close to a small AP to
control how the transmitted/received signals are interacting with the propagation environment and focused in
different directions depending on where the users are
located for the moment. Such an IRS can be thought of
as an additional antenna array that helps the AP to cover
a geographical region.
_ii)_ A large IRS can be deployed in a room where prospective users are located so it has a good channel to both



_≤_ _[P]_

_σ_ [2]



��� ~~�~~ _β_ d + _N_ �



��� ~~�~~



� 2
_β_ IRS ~~�~~ _,_ (3)
�



where the upper bound is achieved when all the terms inside the absolute value have the same phase, which can be
obtained by tuning the phase-delays of the IRS elements. In
particular, we need to set _φ_ [IRS] _n_ = _ψ_ _n_ [TX] + _ψ_ _n_ [RX] _−_ _ψ_ d so that
each of the scattered paths has a phase-delay that matches
with the direct path. In a more advanced setup with multiple
antennas at the transmitter and/or receiver, the reflection
coefficients (e.g., phase-delays) can be jointly optimized with
transmitters and receivers to maximize performance metrics
such as the spectral efficiency or the energy efficiency of the
end-to-end link [161].
The benefit of adding an IRS to a single-antenna communication link is illustrated in Fig. 10, where the spectral efficiency log 2 (1 + SNR) is used as performance metric. The end-to-end channel gain via each IRS element is
_β_ IRS = _−_ 150 dB, which corresponds to having LoS links
with _−_ 75 dB from the transmitter to the IRS and from the IRS

to the receiver. We then consider two cases: _1)_ a weak direct
path with _β_ d = _−_ 100 dB, and _2)_ a strong direct path with
_β_ d = _−_ 75 dB. The transmit power is selected to correspond
to 10 mW per 20 MHz. Fig. 10 shows the spectral efficiency
as a function of the number of IRS elements _N_ ; the baseline
without any IRS is also shown for reference. We notice that
the IRS greatly improves the performance when the direct
path is weak (Case 1), while the performance gains are more
modest when the direct path is strong (Case 2). In any case,
we need hundreds of elements for the IRS to be effective; the
end-to-end channel gain of the path via a single IRS element
is typically very small but the phase-aligned combination
of many such paths can give considerable gains. The large
number of elements might seem like a showstopper but it is
not since the elements are spread out over two dimensions



VOLUME 4, 2021 19


**FIGURE 11.** An IRS takes an incoming wave and reflects it as a beam in a
particular direction or towards a spatial point.


-70





-72


-74


-76


-78


-80
0 5 10 15 20 25 30 35 40 45


**FIGURE 12.** End-to-end channel gain between a transmitter and receiver that
are 45 m apart depends on where the IRS is deployed. It is largest when the
IRS is either close to the transmitter or the receiver.


the AP and the UE, even if the direct path between them
happens to be weak.


In addition to increasing the signal strength of a single
user, an IRS can improve the channel rank (for both singleand multi-user MIMO), suppress the interference [166], and
enhance the multicasting performance [167]. Essentially anything that can be done with traditional beamforming can also
be implemented using an IRS, and they can be deployed as
an add-on to many existing systems. The beamwidth of the
scattered signal from the IRS will also match that of an equalsized antenna array. Other prospective use cases are cognitive radio [168], [169], wireless power transfer [170]–[172],
physical layer security [173]–[176], NOMA [177], [178], vehicular networks with predictive mobility [164], [179], [180],
coordinated multipoint transmission (CoMP) [181], UAV
communications [182]–[186] and backscattering, where IoT
devices near the surface can communicate with an AP at

zero energy cost [153]. It still remains to identify which
use cases lead to the largest improvements over the existing
technology [155].
The main open research challenges are related to the
signal processing, hardware implementation, channel modeling, experimental validation, and real-time control. Al


N. Rajatheva _et al._ : Scoring the Terabit/s Goal: Broadband Connectivity in 6G


though software-controllable meta-surfaces exist [153], the
accuracy of the reconfigurability is currently limited due to,
for example, the small number of possible phase values per
element, fixed amplitudes for each phase delay [187], and by
only having control over groups of elements. Although semiaccurate physical channel models exist for LoS scenarios,
these omit mutual coupling and other hardware effects that
are inevitable in practice. Therefore, experimentally validated channel models are strongly needed. The performance
loss caused by using practical low-resolution hardware (such
as 1-bit delay resolution) also needs to be carefully studied,
especially for the case when a large number of reflecting
elements is required to achieve a considerable beamforming
gain [188], [189]. Furthermore, wireless networks, in general, operate in broadband channels with frequency selectivity. As such, the phase-delay pattern of the IRS needs to strike
a balance between the channels of different frequency subbands, which further complicates the joint active and passive
beamforming optimization. Besides, most of the existing
works on IRS assume that the maximum amount of power
is scattered by the IRS, while only the IRS phase shifts are
fine-tuned to improve the system performance. However, it
remains unknown in which circumstances one can achieve

higher performance by jointly optimizing the amplitudes and
phase shifts, and how to balance the performance gain and
practical hardware cost as well as complexity [190], [191].
Lastly, the control interface and the channel estimation are
difficult when using a passive surface that cannot send or
receive pilot signals. Hence, channel measurements can only
be made from pilots sent by other devices and received at
other locations.

The radio resources required for the channel estimation
and control will grow linearly with the number of elements
and may become huge in cases of interest unless a clever
design that utilizes experimentally validated channel models
can be devised. To reduce the channel estimation overhead

and implementation complexity, two-timescale beamforming
optimization can be a potential approach [192]. An anchorassisted two-phase channel estimation scheme is proposed
in [193], where two anchor nodes are deployed near the
IRS for helping the BS to acquire the cascaded BS-IRS-user
channels. The power consumption of the control interface
will likely dominate the total energy consumption of an IRS
since there are no power amplifiers or RF chains [155].
One channel estimation approach is to repeatedly transmit
pilot signals and change the IRS configuration according
to a codebook to identify the preferred operation [194].
Another potential solution is to have a small number of
active elements in the IRS and then use deep learning or
compressive sensing to extrapolate the estimates made in
those element to an estimate of the entire channel [195].
Finally, proper deployment of the IRS with active BSs in
such hybrid networks to maximize the system performance
is another important problem to solve, especially in practical
multi-cell scenarios. The design considerations include the
rank of the channel, the beamforming gain, LoS versus non


20 VOLUME 4, 2021


N. Rajatheva _et al._ : Scoring the Terabit/s Goal: Broadband Connectivity in 6G


LoS propagation, and so on.
Recently, IRSs and their various equivalents have also
drawn significant attention from the industry [167]. In
November 2018, NTT DoCoMo and Metawave jointly
demonstrated that by properly deploying a meta-structure
based reflecting surface, 560 Mbps communication rate can
be achieved in the 28 GHz band as compared to 60 Mbps
without it. There are startups such as Greenerwave, Echodyne, Kymeta that attempt to commercialize IRS-type technologies for consumer-grade use cases.


_**C. SCALABLE CELL-FREE NETWORKING WITH**_

_**USER-CENTRIC ASSOCIATION**_

The large performance disparity between cell-center and celledge UEs is one of the main drawbacks of the traditional
cellular network topology. The concept of cell-free massive
MIMO has recently emerged to overcome this long-standing
issue by providing consistent performance and seamless
handover regardless of the UEs’ positions [196]–[199]. By
conveniently combining elements from massive MIMO,
small cells, and user-centric CoMP with joint transmission/reception [200]–[204], cell-free massive MIMO gives
rise to a cell-less architecture characterized by almost uniform achievable rates across the coverage area. For this
reason, it is widely regarded as a potential physical-layer
paradigm shift for 6G wireless systems.
In cell-free massive MIMO, conventional APs equipped
with massive co-located antenna arrays are replaced by a
large number of low-cost APs equipped with few antennas,
which cooperate to jointly serve the UEs. To enable such
cooperation, the APs are connected to one or more CPUs
via fronthaul links, as illustrated in Fig. 13. Distributing the
transmit/receive antennas over a large number of cooperating
APs has a two-fold advantage over cellular massive MIMO.
First of all, it enables each UE to be located near one or a few
APs with high probability and, as a consequence, to be jointly
served by a reasonable number of favorable antennas with
reduced path loss. In addition, as the serving antennas for
each UE belong to spatially separated APs (which are usually
seen with different angles), it brings an improved diversity
against blockage and large-scale fading.
Unlike CoMP with joint transmission/reception, which is
traditionally implemented in a network-centric fashion with
well-defined edges between clusters of cooperating APs, cellfree massive MIMO adopts a user-centric approach, where
the above clusters are formed so that each UE is served

by its nearest APs [201], [204], [205]. Furthermore, as for
CoMP systems, the cell-free approach greatly benefits from
TDD operations, which allow to use uplink pilot signals for
both upink and downlink channel estimation. While CoMP
was designed as an add-on to an existing cellular network,
the deployment architecture and protocols are co-designed in
cell-free massive MIMO to deliver uniform service quality

[196]–[199].
Let us consider a cell-free massive MIMO deployment
with single-antenna UEs, where MMSE channel estimation is



**TABLE 3.** Symbols Definition for a Cell-Free Massive MIMO Deployment

|Symbol|Description|
|---|---|
|_K_|~~Number of UEs in the system~~<br>|
|_M_|~~Number of APs~~<br>|
|_N_A_P_|~~Number of antennas at each AP~~<br>|
|_βk,m_|~~scalar coefﬁcient representing large-scale fading between~~<br>the_ k_th UE and the_ m_th AP<br>|
|**g**_k,m_<br><br><br>|_N_A_P_~~ -dimensional uplink channel between the~~_ k_~~th UE~~<br>and the_ m_th AP<br>|
|_σ_~~2~~<br>_z,k, σ_~~2~~<br>_w,_|_m_ ~~Receiver noise power at the~~_ k_~~th UE and~~_ m_~~th AP~~<br>|
|_τc_<br>|~~Length, in discrete-time samples, of the channel coherence~~<br>block<br>|
|_τd_~~ (~~_τu_~~)~~|~~Length, in discrete-time samples, of the downlink (uplink)~~<br>data transmission phase<br>|
|_τp_|~~Length of and number of mutually orthogonal pilot se-~~<br>quences<br>|
|_Pk_|~~Set of UEs using the same pilot sequence as the~~_ k_~~th UE~~<br>|
|_ηk_|~~Uplink power for the~~_ k_~~th UE during channel training~~<br>phase<br>|
|_η_~~D~~~~_L_~~<br>_k,m_|~~Power used on the downlink by the~~_ m_~~th AP to transmit~~<br>data to the_ k_th UE<br>|
|_η_~~U~~~~_L_~~<br>_k_|~~Uplink power for the~~_ k_~~th UE during uplink data transmis-~~<br>sion phase<br>|
|_Mk_|~~Set of APs serving the~~_ k_~~th UE~~<br>|
|_Km_|~~Set of users served by the~~_ m_~~th AP~~|



performed based on the transmission of uplink pilots, maximum ratio transmission/combining is used at the APs, and the
small-scale fading coefficients are i.i.d. complex Gaussian
distributed (i.e., the usual Rayleigh fading assumption). Considering the notation reported in Table IV-C, a lower bound
on the uplink and downlink capacities for the _k_ th UE can be
expressed as in (4) and (5) at the top of the page, which are
based on the use-and-then-forget bounding technique [135].
In these expressions, _B_ represents the signal bandwidth,
whereas _σ_ _z,k_ [2] [and] _[ σ]_ _w,m_ [2] [are the receiver noise powers at the]
_k_ th UE and at the _m_ th AP, respectively. Moreover, the meansquare of the channel estimates is

_η_ _k_ _β_ _k_ [2] _,m_
_γ_ _k,m_ = _._ (6)
~~�~~ _η_ _i_ _β_ _i,m_ + _σ_ _z,k_ [2]

_i∈P_ _k_


The expressions in (4) and (5) are general enough to
describe the performance of three relevant systems.

_i)_ Letting _K_ _m_ = _{_ 1 _,_ 2 _, . . ., K}, ∀m_ yields an FCF association, where all APs transmit to all the UEs in the
network. This is the originally conceived version of
cell-free massive MIMO, which is practically applicable
only to a system deployed in a limited area [205].
_ii)_ A more practically relevant scenario is obtained by
using a CF-UC association, where each UE is served
only by a limited number of APs [199], [204], [205].
The optimal UE-AP association is a complex combinatorial problem that depends on the type of the
adopted precoding and hardware constraints. However,
one viable and simple solution is to let each UE be
served only by a predetermined number of APs, which
can be selected based on the large-scale fading coefficients. For example, the _k_ th user can be served by



VOLUME 4, 2021 21


N. Rajatheva _et al._ : Scoring the Terabit/s Goal: Broadband Connectivity in 6G



� 2



_N_ A _P_



�� _m∈M_ _k_



~~�~~ _η_ _k,m_ [D] _[L]_ _[γ]_ _[k,m]_



~~�~~



_R_ [D] _k_ _[L]_ = _B_ _[τ]_ _[d]_ log 2

_τ_ _c_
















1 +



_,_ (4)



+ _σ_ _z,k_ [2]



2



~~~~





~~~~



�

_j∈K_



� _η_ _j,m_ [D] _[L]_ _[β]_ _[k,m]_ [+] �

_m∈M_ _j_ _j∈P_ _k_



�



� _N_ A _P_

_j∈P_ _k_ _\{k}_



_m∈M_ _j_



~~�~~ _η_ _j,m_ [D] _[L]_ _[γ]_ _[k,m]_
















~~~~



 _m_ [�] _∈M_



� 2



_η_ _k_ [U] _[L]_ _[N]_ [A] _[P]_



_γ_ _k,m_

�� _m∈M_ _k_



_R_ [U] _k_ _[L]_ = _B_ _[τ]_ _[u]_ log 2

_τ_ _c_







1 +






+ � _σ_ _w,m_ [2] _[γ]_ _[k,m]_

_m∈M_ _k_



2

_β_ _j,m_
_β_ _k,m_ ~~�~~



� _η_ _j_ [U] _[L]_ �

_j∈K_ _m∈M_



�



� _β_ _j,m_ _γ_ _k,m_ + �

_m∈M_ _k_ _j∈P_ _k_



� _N_ A _P_ _η_ _j_ [U] _[L]_

_j∈P_ _k_ _\{k}_













(5)



� _γ_ _k,m_
~~�~~ _m∈M_ _k_



~~_η_~~ _j_
~~�~~ _η_ _k_



**FIGURE 13.** A cell-free massive MIMO system consists of distributed APs that jointly serve the UEs. The cooperation is facilitated by a fronthaul network and a
CPU.



the _N_ U _C_ APs with the best average channel conditions.
Let _O_ _k_ : _{_ 1 _, . . ., M_ _} →{_ 1 _, . . ., M_ _}_ denote the
sorting operator for the vector [ _β_ _k,_ 1 _, . . ., β_ _k,M_ ], such
that _β_ _k,O_ _k_ (1) _≥_ _β_ _k,O_ _k_ (2) _≥_ _. . . ≥_ _β_ _k,O_ _k_ ( _M_ ) . The set
_M_ _k_ of the _N_ U _C_ APs serving the _k_ th UE is then given
by


_M_ _k_ = _{O_ _k_ (1) _, O_ _k_ (2) _, . . ., O_ _k_ ( _N_ U _C_ ) _}._ (7)


Consequently, the set of UEs served by the _m_ th AP is
defined as _K_ _m_ = _{k_ : _m ∈M_ _k_ _}_ .
_iii)_ Assuming a small number of APs with a large number
of antennas each and supposing that each UE can be
associated to only one AP (so that the sets _M_ _k_ _, ∀k_ have
cardinality 1) yields a traditional multi-cell mMIMO
deployment.



Some numerical results illustrating the performance of
the FCF, CF-UC and mMIMO deployments are reported in
Figs. 14-15 and in Tables 4-5. The simulation setup is the
following: communication bandwidth _B_ = 20 MHz; carrier
frequency _f_ 0 = 1 _._ 9 GHz; antenna height at the APs 10 m
and at the UEs 1 _._ 65 m; thermal noise with power spectral
density _−_ 174 dBm/Hz; front-end receiver at the APs and at
the UEs with noise figure of 9 dB. The number of APs is
_M_ = 100 and each has _N_ A _P_ = 4 antennas and the number of
UEs simultaneously served in the system on the same timefrequency coherence block is _K_ = 30. The APs and UEs are
deployed at random positions on a square area of 1000 _×_ 1000
square meters. The large-scale fading coefficients _β_ _k,m_ are
evaluated as in [206], [207]. In order to avoid boundary
effects, the square area is wrapped around [197], [199]. The



22 VOLUME 4, 2021


N. Rajatheva _et al._ : Scoring the Terabit/s Goal: Broadband Connectivity in 6G



mutually orthogonal pilot sequences have length _τ_ _p_ = 16;
the downlink and uplink data transmission phases of each
coherence block contain _τ_ _u_ = _τ_ _d_ = _[τ]_ _[c]_ _[−]_ 2 _[τ]_ _[p]_ samples, where

_τ_ _c_ = 200 is the length of the coherence block in samples. The
uplink transmit power for channel estimation is _η_ _k_ = _τ_ _p_ _p_ _k_,
with _p_ _k_ = 100 mW, _∀k_ = 1 _, . . ., K_ . With regard to power
control, the results have been obtained for the cases of PPA
and FPA-DL for the downlink. For the uplink, instead, UPAUL and FPA-UL has been considered. More precisely, for the
downlink, denoting as _P_ m [D] _ax,AP_ _[L]_ [the maximum power avail-]
able at each AP, the power coefficients for the PPA are set
as _η_ _k,m_ [D] _[L]_ [=] _[ γ]_ _[k,m]_ _[P]_ [ D] m _ax,AP_ _[L]_ _[/]_ [(][�] _k∈K_ ( _m_ ) _[γ]_ _[k,m]_ [)][ and for the FPA-]

DL rule as _η_ _k,m_ [D] _[L]_ [=] _[ γ]_ _k,m_ _[−]_ [(] _[α]_ [D] _[L]_ [+1)] _P_ m [D] _ax,AP_ _[L]_ _[/]_ [(][�] _k∈K_ ( _m_ ) _[γ]_ _k,m_ _[−][α]_ [D] _[L]_ ),

with _α_ D _L_ = _−_ 0 _._ 5. On the uplink, instead, denoting by _P_ m [U] _ax_ _[L]_
the maximum power available at each UE, for the UPA-UL
we let _η_ _k_ [U] _[L]_ = _P_ m [U] _ax_ _[L]_ [,] _[ ∀]_ _[k]_ [ = 1] _[, . . ., K]_ [, while for the FPA-UL]
we have _η_ _k_ [U] _[L]_ = min � _P_ m [U] _ax_ _[L]_ _[,][ P]_ [0] _[γ]_ [¯] _k_ _[−][α]_ [U] _[L]_ �, _∀_ _k_ = 1 _, . . ., K_,



_k∈K_ ( _m_ ) _[γ]_ _[k,m]_ [)][ and for the FPA-]



the uplink. In order to give a deeper look at the comparison
between FCF and CF-UC deployments, in Fig. 15 we report
the performance of the FCF deployment and of the CF-UC
deployment for several values of _N_ _UC_ . Results show that
taking _N_ _UC_ _≥_ 10 provides performance levels comparable
or superior to the FCF deployment. Since the FCF deployment corresponds to an CF-UC deployment with _N_ _UC_ = _M_,
the figure shows that there is an “optimal” value of _N_ _UC_
that maximizes the system performance. Besides this, the
main conclusion that can be drawn from this figure is that
CF-UC deployments offer performance levels comparable or
better than FCF deployments, but with much lesser signaling
and backhaul overhead, so special care should be devoted to
the design of the UE-AP association rules since they have a
crucial impact on the system performance.
The cell-free massive MIMO architecture is not meant to

increase the peak rates in broadband applications, since these
can only be achieved in extreme cases, but has been shown to
vastly outperform traditional small-cell and cellular massive
MIMO for the majority of UEs [197], [198], [208]. A cellfree massive MIMO deployment can also provide support
for the implementation of low-latency, mission-critical applications. The availability of many APs, coupled with the
rapidly decreasing cost of storage and computing capabilities, permits using cell-free massive MIMO deployments for
the caching of content close to the UEs and for the realization
of distributed computing architectures, which can be used to
offload network-intensive computational tasks. Moreover, in
low-demand situations, some APs can be partially switched
off (control signals may still be transmitted) with a limited
impact on the network performance, thus contributing to
reducing the OPEX of network operators and their carbon
footprint.


1) Cell-Free Initial Access
The basic connection procedures of cell search and random
access determine the network performance in terms of latency, energy consumption, and the number of supported
UEs [209]. These basic functionalities are currently tailored
to the cellular architecture. Fig. 16 illustrates initial access
based on different mechanisms. As shown in the figure, in the
cellular system, a UE attempts to access a cell in a cellular
network. The UE usually chooses the strongest cell based
on the measurement of received synchronization signals and
will be subject to interference from neighboring cells. On
the contrary, in a cell-free system, a UE attempts to access
a cell-free network where all the neighboring APs support
the UE’s access to the network. To enable this, the traditional
cell identification procedure must be re-defined, including
the synchronization signals and how system information
is broadcast. Similarly, a new random access mechanism
suitable for cell-free networks is needed such that some

messages in the random access procedure can be transmitted
and processed at multiple APs.
In summary, it is desirable that idle/inactive UEs can
harness the benefits of the cell-free architecture as much as



DL rule as _η_ _k,m_ [D] _[L]_ [=] _[ γ]_ _k,m_ _[−]_ [(] _[α]_ [D] _[L]_ [+1)] _P_ m [D] _ax,AP_ _[L]_ _[/]_ [(][�]



with ¯ _γ_ _k_ = � ~~�~~ _m∈M_ _k_ _[γ]_ _[k,m]_ [. The performance of the FCF]

and CF-UC are compared with a mMIMO system with 4
APs and 100 antennas each covering the same area. The same
transceiver signal processing is assumed for the APs and the
APs, i.e., the performance of the mMIMO, FCF and CF-UC
systems are compared assuming MR processing in both uplink and downlink. For the mMIMO system, in order to consider a fair comparison, we set _P_ m [D] _ax,AP_ _[L]_ [=] _[ MP]_ [ D] m _ax,AP_ _[L]_ _[/]_ [4]
and present results for the PPA-DL and for the UPA-DL, with
the power coefficients set as _η_ _k,m_ [D] _[L]_ [=] _[ P]_ [ D] m _ax,AP_ _[L]_ _[/][ |K]_ _[m]_ _[|][ γ]_ _[k,m]_ [.]
We assume, _P_ m [D] _ax,AP_ _[L]_ [= 200][ mW,] _[ P]_ [ D] m _ax,AP_ _[L]_ [= 5][ W, and]
_P_ m [U] _ax_ _[L]_ [= 100][ mW.]
Fig. 14 reports the CDF of the rate per UE for the three
considered deployments. It is clearly seen that mMIMO,
while providing very large rates to the lucky UEs that are
very close to the macro AP site (this corresponds to the
upper-right part of the figure), is largely outperformed by
FCF and CF-UC deployments when considering the vast
majority of all the UEs. Indeed, the figure reveals that about
90% of the UEs on the downlink and 80 % of the UEs

on the uplink enjoy much better rates with FCF and CFUC deployments than with mMIMO. This situation is again
clearly represented by the numbers reported in Tables 4-5,
where we have reported the 1%, 5%, 50% and 90% likely
per-user rates. It is indeed seen that mMIMO outperforms
novel FCF and CF-UC deployments only when considering
the 90%-likely per user rate, while being practically not-able
to serve a good share of the UEs. [3]

Fig. 14 shows the performance of FCF versus a CF-UC
deployment where each UE is connected to the _N_ _UC_ = 10
APs with the largest large-scale fading coefficients. It is seen
that the two systems perform quite similarly, especially in


3 The reader should however not be led to draw the conclusion that
“massive MIMO does not work”. Indeed, we are considering here an extreme
situation with simple MR processing and with a large number of UEs being
served on the coherence block. In a less loaded scenario, with a smaller
number of UEs and more advanced processing schemes such as regularized
zero-forcing beamforming [135], the performance of mMIMO is restored
and the gap with FCF and CF-UC deployments gets reduced.



VOLUME 4, 2021 23


1


0.8


0.6


0.4


0.2


0



|Col1|Col2|Col3|Col4|Col5|
|---|---|---|---|---|
||||||
||||||
|||FCF, PP<br>FCF, FP<br>CF-UC,<br>|A<br>A-DL<br>PPA<br>||
|||CF-UC,<br>mMIMO<br>mMIMO|FPA-DL<br>, UPA-DL<br>, PPA|FPA-DL<br>, UPA-DL<br>, PPA|


0 10 20 30 40 50

DL rate per user [Mbps]



N. Rajatheva _et al._ : Scoring the Terabit/s Goal: Broadband Connectivity in 6G


1


0.8


0.6



|Col1|Col2|Col3|Col4|Col5|Col6|Col7|
|---|---|---|---|---|---|---|
||||||||
||||||||
|||||FCF, UP<br>FCF, FPA<br>CF-UC, U<br>|A-UL<br>-UL<br>PA-UL<br>||
||||||||
||||||||
|||||CF-UC, F<br>mMIMO,<br>mMIMO,|PA-UL<br> UPA-UL<br> FPA-UL|PA-UL<br> UPA-UL<br> FPA-UL|
||||||||
||||||||
||||||||


0 10 20 30 40 50

UL rate per user [Mbps]



0.4


0.2


0



**FIGURE 14.** Cumulative distribution functions (CDFs) of the downlink and uplink rate per UE of the FCF, CF-UC and mMIMO with different power allocation
strategies. Parameters: _M_ = 100, _N_ A _P_ = 4, _K_ = 30, _N_ _UC_ = 10, _τ_ _p_ = 16, _α_ D _L_ = _−_ 0 _._ 5, _P_ 0 = _−_ 10 dBm, and _α_ U _L_ = 0 _._ 5.


**TABLE 4.** Performance of FCF, CF-UC and mMIMO with UPA-UL and PPA in Downlink. Parameters: _M_ = 100, _N_ A _P_ = 4, _K_ = 30, _N_ _UC_ = 10, _τ_ _p_ = 16

|Col1|DL 1%-rate DL 5%-ra|te DL 50%-rate|DL 90%-rate UL|1%-rate|UL 5%-rate|UL 50%-rate|UL 90%-rate|
|---|---|---|---|---|---|---|---|
|~~FCF~~<br>|~~7.38 Mbps~~<br>~~13.9 Mbps~~<br><br>|~~27.4 Mbps~~<br>|~~35.2 Mbps~~<br>~~0.4~~<br><br>|~~7 Mbps~~<br>|~~2.1 Mbps~~<br>|~~21.3 Mbps~~<br>|~~28.5 Mbps~~<br>|
|~~CF-UC~~<br>|~~5.8 Mbps~~<br>~~12.1 Mbps~~<br><br>|~~26.6 Mbps~~<br>|~~34.7 Mbps~~<br>~~0.4~~<br><br>|~~2 Mbps~~<br>|~~2 Mbps~~<br>|~~21.5 Mbps~~<br>|~~29.2 Mbps~~<br>|
|~~mMIMO~~|~~0 Mbps~~<br>~~0 Mbps~~|~~0.2 Mbps~~|~~38.6 Mbps~~<br>~~0.0~~|~~004 Mbps~~|~~0.02 Mbps~~|~~6.3 Mbps~~|~~46.1 Mbps~~|



**TABLE 5.** Performance of FCF, CF-UC with FPA-DL and mMIMO with UPA-DL and FPA-UL. Parameters: _M_ = 100, _N_ A _P_ = 4, _K_ = 30, _N_ _UC_ = 10, _τ_ _p_ = 16,
_α_ D _L_ = _−_ 0 _._ 5, _P_ 0 = _−_ 10 dBm, and _α_ U _L_ = 0 _._ 5


|Col1|DL 1%-rate DL 5%-ra|te DL 50%-rate|DL 90%-rate UL|1%-rate|UL 5%-rate|UL 50%-rate|UL 90%-rate|
|---|---|---|---|---|---|---|---|
|~~FCF~~<br>|~~10.2 Mbps~~<br>~~15.6 Mbps~~<br><br>|~~25 Mbps~~<br>|~~30.8 Mbps~~<br>~~1.9~~<br><br>|~~ Mbps~~<br>|~~5.6 Mbps~~<br>|~~21.6 Mbps~~<br>|~~26.2 Mbps~~<br>|
|~~CF-UC~~<br>|~~10 Mbps~~<br>~~15.9 Mbps~~<br><br>|~~12 Mbps~~<br>|~~33.4 Mbps~~<br>~~1.7~~<br><br>|~~ Mbps~~<br>|~~5.2 Mbps~~<br>|~~21.6 Mbps~~<br>|~~28.5 Mbps~~<br>|
|~~mMIMO~~|~~0.01 Mbps~~<br>~~0.09 Mbps~~|~~2.6 Mbps~~|~~23.3 Mbps~~<br>~~0.0~~|~~02 Mbps~~|~~0.1 Mbps~~|~~8.3 Mbps~~|~~41.2 Mbps~~|



1


0.8


0.6


0.4


0.2


0



|.55|Col2|Col3|Col4|
|---|---|---|---|
|.45<br>0.5||||
|22<br>|24<br>26|||
|||||
|||||


0 5 10 15 20 25 30 35 40

Average DL rate per user [Mbps]



0.6


0.4


0.2


0



1


|0.52<br>0.5|Col2|Col3|Col4|Col5|
|---|---|---|---|---|
|0.5<br>0.52|||||
|0.48<br>|||||
|0.48<br>|||||
|20.5<br>21<br>|20.5<br>21<br>|21.5<br>22|||
||||||
||||||



0 10 20 30 40

Average UL rate per user [Mbps]



0.8











**FIGURE 15.** Cumulative distribution functions (CDFs) of the downlink and uplink rate per UE of the CF-UC with different values of _N_ _UC_ compared with the FCF
with FPA-DL and FPA-UL. Parameters: _M_ = 100, _N_ A _P_ = 4, _K_ = 30, _τ_ _p_ = 16, _α_ D _L_ = _−_ 0 _._ 5, _P_ 0 = _−_ 10 dBm, _α_ U _L_ = 0 _._ 5.



the active UEs can. To realize this, it is imperative to redesign
the procedures of cell search and random access. To reduce
the latency and improve the resource utilization efficiency,
NOMA-enabled two-step random-access channel [210] or
autonomous grant-free data transmission [211] should be
investigated in cell-free networks.



2) Implementation Challenges


The huge amount of CSI that needs to be exchanged
over the fronthaul to implement centralized joint precoding/combining is a long-standing scalability bottleneck
for the practical implementation of CoMP and network
MIMO [201], [212]. In its original form [197], [198], cellfree massive MIMO avoids CSI exchange by only perform


24 VOLUME 4, 2021


N. Rajatheva _et al._ : Scoring the Terabit/s Goal: Broadband Connectivity in 6G


**FIGURE 16.** Initial access based on cellular system (left) and cell-free system (right).



ing signal encoding/decoding at the CPU, while combining/precoding (such as matched filtering, local zero-forcing,
and local MMSE processing) is implemented at each AP
using locally acquired CSI. By synchronizing the APs, the
signals can be coherently combined without the need for
CSI sharing. In this case, each AP consists of antennas
and UE-grade RF modules that perform digital operations
such as channel estimation, combining/precoding, interpolation/decimation, digital pre-distortion, and DFT [202].
However, the performance of cell-free massive MIMO
systems can be sensibly boosted by increasing the level of coordination among the APs, in particular, by sending the CSI
to the CPU and performing joint combining/precoding [208],

[213] as in conventional CoMP and network MIMO. In
this case, the APs can be made smaller since most of the
digital operations are carried out at the CPU. While the data
rates can be increased, the drawback is higher computational
complexity and latency, and the fronthaul signaling might
also increase. If there are 64 antennas taking 20-bit samples,
the total fronthaul capacity will far exceed 1 Gbps at the sampling rates of interest. This imposes significant challenges for
interconnection technologies, such as CPRI and on the I/O
interface of processing circuits [214]. A potential solution
to this problem could be to form separate antenna clusters
and have a separate CPRI for each cluster. However, this
increases the overall complexity of the system. Over-theair bi-directional signaling between the APs and the UEs
might be utilized as a flexible alternative to fronthaul signaling [215]–[217]. Furthermore, it is necessary to investigate
scalable device coordination and synchronization methods to
implement CSI and data exchange [205].
To provide the aforementioned gains over cellular technology, cell-free massive MIMO requires the use of a large number of APs and the attendant deployment of suitable fronthaul
links. Although there are concepts (e.g., radio stripes where
the antennas are integrated into cables [202]) to achieve
practically convenient deployment, the technology is mainly
of interest for crowded areas with a huge traffic demand or
robustness requirements. A cell-free network will probably



underlay a traditional cellular network, and is likely to use
APs with large co-located arrays. The potential offered by
integrated access and backhaul techniques could also be very
helpful in alleviating the fronthaul problem and in reducing
the cost of deployment.
Cell-free massive MIMO can be deployed in any frequency
band, including below-6 GHz, mmWave, sub-THz, and THz
bands. In the latter cases, the APs can serve each UE using
a bandwidth of 100 GHz or higher, which yields extremely
high data rates over short distances and low mobility. The
spatial diversity gains of the cell-free architecture become
particularly evident in such scenarios because the signal from
a single AP is easily blocked, but the risk that all neighboring
APs are simultaneously blocked is vastly lower.


_**D. INTEGRATED ACCESS AND BACKHAUL**_

Considering dense networks, we need to provide multiple
APs of different types with backhaul connections. There are
different backhauling methods today, among which wireless
microwave and optical fiber are the dominant ones. Fiber
provides reliable transport with demonstrated Tbps rates. For
this reason, the use of (dark) fiber will continue growing
in 5G and beyond. However, fiber deployment requires a
noteworthy initial investment for trenching/installation, may
incur long installation delays, and even may be not allowed
in some metropolitan areas.
Wireless backhaul using microwave is a competitive alternative to fiber, supporting up to 100 Gbps [218]. Particularly,
compared to fiber, microwave backhaul is an economical
and scalable option, with considerably lower cost and flexible/timely deployment (e.g., no digging, infrastructure displacement, and possible to deploy in principle everywhere)

[219], [220]. Today, microwave backhauling operates mainly
in the 4–70/80 GHz range. Then, given the fact that 6G
networks will see an even greater level of densification and
spectrum heterogeneity, compared to 4G and even 5G networks, wireless backhauling is expected to play a major role,
especially at mmWave and THz carrier frequencies [220].
Following the same reasoning, IAB networks, where the



VOLUME 4, 2021 25


**FIGURE 17.** An example of a multi-hop IAB network.


operator uses part of the spectrum resources for wireless
backhauling, has recently received considerable attention

[221], [222]. IAB aims to provide flexible low-cost wireless
backhaul using 3GPP NR technology in IMT bands, and
provide not only backhaul but also the cellular services in the
same node. This will be a complement to existing microwave
point-to-point backhauling in suburban and urban areas (see
Fig. 17).
Wireless backhaul was initially studied in 3GPP in the
scope of LTE relaying [223]. However, there have been only a
few commercial LTE relay deployments with separate bands
for access and backhaul as well as only one-hop relaying.
This is mainly because the existing LTE spectrum is very
expensive to be used for backhauling, and also network
densification did not reach the expected potential in the 4G
timeline. As opposed, IAB is expected to be more commercially successful, compared to LTE relaying, mainly because:


_•_ The large bandwidth available in mmWave (and, possibly, higher) frequencies creates more economically
viable opportunities for backhauling.

_•_ The limited coverage of high frequencies creates a
growing demand for AP densification, which, in turn,
increases the backhauling requirement.

_•_ Advanced spatial processing features, such as massive
MIMO, enables the same bandwidth to be simultaneously used for both access and backhaul [135].


1) On the Performance of IAB Networks

IAB supports both sub-6 GHz and mmWave spectrum as well
as both inband and outband backhauling, where the wireless
backhaul links operate, respectively, in the same and different
frequency bands, as the access links. In this subsection,
we present a comparison between the performance of IAB
and fiber-connected networks using mmWave spectrum and
inband backhauling.
Assume an outdoor two-tier HetNet with multiple MBSs
(M: macro), SBSs (S: small) and UEs. Here, both the MBSs
(IAB donor, in the 3GPP terminology) and the SBSs (IAB



N. Rajatheva _et al._ : Scoring the Terabit/s Goal: Broadband Connectivity in 6G


node, in the 3GPP terminology) use wireless connections for
both backhaul and access. Moreover, only the IAB donors
are fiber-connected while the IAB nodes, i.e., SBSs, receive
backhaul data from the IAB donors wirelessly.
We model the network as an FHPPP, e.g., [224], [225],
in which the MBSs, the SBSs and the UEs are randomly
distributed within a finite region according to mutuallyindependent FHPPPs having densities _φ_ M, _φ_ S and _φ_ U, respectively. In this way, following the mmWave channel model in

[225], the power received by each node can be expressed as


_P_ r = _P_ t _h_ t,r _G_ t,r _L_ (1m) _L_ t,r _||_ **x** t _−_ **x** r _||_ _[−]_ [1] _._ (8)


where **x** t _,_ **x** r are the locations of the transmitter and receiver,
respectively. Moreover, _h_ _t,r_ denotes the small-scale fading,
_P_ t is the transmit power, and _G_ _t,r_ represents the combined
antenna gain of the transmitter and the receiver of the considered link. Also, _L_ _t,r_ and _L_ (1 _m_ ) are the path loss due to
propagation and the reference path loss at 1 meter distance,
respectively. In the simulations, we model the small-scale
fading by a normalized Rayleigh random variable. Using the
5GCM UMa close-in model [226], the path loss, in dB, is
given by



with _i, j_ denoting the indices of the transmit and receive
nodes, _ϕ_ representing the angle between them and _θ_ HPBW
being the half power beamwidth of the antenna. Also, _G_ 0
denotes the directional antenna’s maximum gain and _g_ ( _ϕ_ )
gives the side lobe gain.
In our analysis, the inter-UE interference is neglected,
motivated by the low power of the devices and with the
assumption of sufficient isolation. Also, motivated by the



_,_ (9)
�



_f_ c
� 1 GHz



_r_
PL = 32 _._ 4 + 10 _α_ log 10
� 1 m



+ 20 log 10
�



with _r_ denoting the propagation distance between the nodes,
_f_ c being the carrier frequency, and _α_ is the path loss exponent. Depending on the blockage, NLoS and LoS links are
affected by different path loss exponents, and the propagation
loss in the link between nodes _i_ and _j_ is obtained by



_L_ i,j =



( _r/_ 1 m) _[α]_ [N] _,_ if NLoS,
(10)
�( _r/_ 1 m) _[α]_ [L] _,_ if NLoS.



For the blockage, we use the germ grain model [227, Chapter
14], which is an accurate model for environments with large
obstacles because it takes the obstacles induced blocking
correlation into account. Here, the blockages are distributed
according to an FHPPP distributed with density _λ_ B in the
same area as the other nodes. Also, all blockings are assumed
to be walls of length _l_ B and orientation _θ_, which is an IID
uniform random variable in [0 _,_ 2 _π_ ).
Modeling the beam pattern as a sectored-pattern antenna
array, the antenna gain between two nodes is obtained by

[225]



_G_ _i,j_ ( _ϕ_ ) =



_G_ 0 _−θ_ HPBW 2
� _g_ ( _ϕ_ )



2 2 (11)

_g_ ( _ϕ_ ) otherwise,



HPBW 2 _≤_ _ϕ ≤_ _[θ]_ [HPBW] 2



26 VOLUME 4, 2021


N. Rajatheva _et al._ : Scoring the Terabit/s Goal: Broadband Connectivity in 6G


140


120


100



80


60


40







20

|Col1|=<br>M|8 km-2, =50<br>U|0 km-2, =5<br>B|00 km-2|Col6|Col7|Col8|
|---|---|---|---|---|---|---|---|
||||All IAB|nodes|nodes|||
|||||||||
||Cove|rage probabilit|= 0.81|||30% of no<br>fiber-backh|des<br>uled|
|||||Targe<br>|Targe<br>|||
|||||Targe<br>|Targe<br>|t rate_R_th= 100<br>|Mbps<br>|
|||||Targe|Targe|rate_ R_th= 80|bps|
|||||Targe|Targe|||


20 30 40 50 60 70 80
Fibered SBS density S-fibered (km [-2] )


**FIGURE 18.** Density of IAB nodes providing the same coverage probability
as in fiber-backhauled networks. The parameters are set to
( _α_ LoS _, α_ NLoS ) = (2 _,_ 3), _l_ B = 5 m, _f_ c = 28 GHz, bandwidth= 1 GHz and
_P_ MBS _, P_ SBS _, P_ UE = (40 _,_ 24 _,_ 0) dBm.


1


0.9


0.8



0.7


0.6


0.5


0.4






|Col1|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|Col10|
|---|---|---|---|---|---|---|---|---|---|
||||||S=100 k|m-2||||
|||||||||||
||Targ<br>Targ|et rate_R_th=<br>et rate_R_=|60 Mbps<br> 80 Mbps|||||S=5|0 km-2|
||Targ|th<br>et rate_R_th=|100 Mbps||U=5|00 km-2,|B=500 km-2|||
|||||||||||
||||Ma|cro BSs only,|M=8 km|-2||||



0 10 20 30 40 50 60 70 80 90 100

_All IAB_ Fraction of fiber backhauled SBSs (%) _All fiber_



**FIGURE 19.** Service coverage probability as a function of percentage of
fiber-backhauled SBSs, ( _α_ LoS _, α_ NLoS ) = (2 _,_ 3), _l_ B = 5 m, _f_ c = 28 GHz,
bandwidth= 1 GHz and _P_ MBS _, P_ SBS _, P_ UE = (40 _,_ 24 _,_ 0) dBm.


high beamforming capability in the IAB-IAB backhaul links,
we ignore the interference and assume them to be noiselimited. Then, the UEs are served by either an MBS or an
SBS following open access strategy and based on the maximum average received power rule. Also, we follow the same
approach as in [221] to determine the bandwidth allocation
in the access and backhaul of each SBS proportional to its
load and the number of UEs in the access link such that the

network coverage probability is maximized.
In Figs. 18 and 19, we compare the performance of IAB
and fiber-connected networks, in terms of coverage probability, i.e., the probability of the event that the UEs’ minimum
target rate requirements are satisfied. Note that, in practice, a
number of SBSs may have access to fiber. For this reason, the
figures also present the results for the cases with a fraction
of SBSs being fiber-connected. In this case, we assume
the fiber-connected SBSs to be randomly distributed, and
adapt the association, the resource allocation rule and the
achievable rates correspondingly.
Fig. 18 shows the required number of IAB nodes that
guarantee the same coverage probability as in the cases with
(partially) fiber-connected SBSs. Moreover, Fig. 19 demonstrates the coverage probability versus the fraction of fiberconnected SBSs. The figure also compares the performance
of the IAB network with the cases having only MBSs. In each



simulation setup, we have followed the same approach as in

[221] to optimize different parameters, such as resource allocation and UE association, such that the coverage probability
is maximized. The details of the simulation parameters are
given in the figures’ captions.
As can be seen in the figures, IAB network increases
the coverage probability, compared to the cases with only
MBSs, significantly (Fig. 19). For a broad range of parameter
settings, the IAB network can provide the same coverage
probability as in the fiber-connected network with relatively
small increment in the number of IAB nodes. For instance,
consider the parameter settings of Fig. 18 and the UEs’ target
rate threshold of 100 Mbps. Then, a fully fiber-connected
network with the SBSs density _λ_ s = 60 km _[−]_ [2] corresponds
to an IAB network with density _λ_ _s_ = 85 km _[−]_ [2] resulting
in coverage probability 0 _._ 81. Also, interestingly, providing
30% of the SBSs with fiber connection reduces the required
density to _λ_ s = 70 km _[−]_ [2], i.e., only 16% increment in the
number of SBSs. Then, as the network density increases,
the relative performance gap of the IAB and fiber-connected
networks decrease (Fig. 18). Finally, it is interesting to note
that our results, which are based on FHPPP, give a pessimistic
performance of the IAB networks. The reason is that, in
practice, the network will be fairly well planned, resulting
in even less gap between the performance of IAB and fiberconnected networks.
Such a small increment in the number of fiber-free SBSs,
i.e., IAB nodes, leads to the following benefits:

_•_ _Network flexibility increment_ : As opposed to fiberconnected networks, where the APs can be installed
only in the places with fiber connection, the IAB nodes
can be installed in different places as long as they
have fairly good connection to their parent nodes. This
increases the network flexibility and the possibility for
topology optimization remarkably.

_•_ _Network cost reduction_ : An SBS is much cheaper than
fiber [4] . For instance, as illustrated in [228, Table 7], in
an urban area the fiber cost is in the range of 20000
GBP/km, while an SBS in 5G is estimated to cost
around 2500 GBP per unit [229]. Moreover, different
evaluations indicate that, for dense urban/suburban areas, even in the presence of dark fiber, the IAB network
deployment is an opportunity to reduce the total cost of
ownership.

_•_ _Time-to-market reduction_ : Fiber laying may take a long
time, because it requires different permissions and labor
work. In such cases, IAB can establish new radio sites
quickly. Consequently, starting with IAB and, if/when
required, replacing it by fiber is expected to be a common setup.

Along with these benefits, different evaluations indicate
that, unless for the cases with suburban areas and moder

4 In general, the fiber cost varies vastly in different regions. However, for
different areas, fiber installation accounts to a significant fraction, in the
order of 80% _,_ of the total network cost



VOLUME 4, 2021 27


ate/high tree foliage, with proper antenna height and network density the performance of the IAB network is fairly
robust to blockage, rain and tree foliage, which introduces
it as a reliable system in different environments. These are
the reasons that different operators have shown interest to
implement IAB in 5G networks [230], [231], and the trend
is expected to increase in beyond 5G/6G networks.


_**E. INTEGRATED SPACE AND TERRESTRIAL**_

_**NETWORKS**_

In parallel with the development of terrestrial mobile systems
such as 5G, another major international effort in wireless
communications is the development of space communications networks which enable global coverage at any time and
from anywhere, such as on the sea, over the air and space,
and in rural and remote areas. The concept of using various
space platforms to perform data acquisition, transmission
and information processing has been around for several tens
of years. However, due to the limited bandwidth and large
transmission delay, existing space networks alone cannot
provide the sufficient capacity and guaranteed quality-ofservice to meet the ever increasing demand for global wireless connectivity. Seamlessly interconnecting space networks
with terrestrial networks to support truly global high-speed
wireless communications will be one of the major objectives
of the 6G wireless systems [232].
Fig. 20 shows the architecture of a typical ISTN which
consists of three layers: the spaceborne network layer, the airborne network layer, and the conventional ground-based network layer. The spaceborne network consists of various orbiting satellites such as the geostationary-Earth orbit (GEO)
satellites, medium-Earth orbit (MEO) and LEO satellites, and
mini satellites known as CubeSats. The airborne network

consists of various aerial platforms including stratospheric
balloons, airships and aircrafts, UAVs, and HAPSs. The
traditional ground-based networks include wireless cellular
networks, satellite ground BSs, mobile satellite terminals,
and many more. The integrated network can make full use
of the signal propagation characteristics of large space coverage, low loss LoS transmission, etc. to achieve seamless
high-speed communications with global coverage.
There are a number of grand technological challenges to
achieve the ISTN with high capacity and low cost. The two
predominant bottlenecks that limit the current ISTN development are the available bandwidth for high-speed aerial
backbones and the spectral efficiency for direct air-to-ground
communications between the airborne and ground based
networks. Aerial backbones uses high-speed links to connect
the major nodes in the airborne network and between the
airborne and ground based networks. They play a pivotal
role in the ISTN by handling the aggregation and distribution
of various data flows such as voice, video, Internet, and
other data sources. Because it sits in-between the spaceborne
network and ground based network, the airborne network
is an indispensable and important intermediate layer in the
ISTN and hence the high-speed, flexible, and all-weather



N. Rajatheva _et al._ : Scoring the Terabit/s Goal: Broadband Connectivity in 6G


mobile aerial backbones become the important infrastructure
to support the services provided by the airborne network. Unfortunately, existing aerial backbones mainly use microwave
links with very limited bandwidth and, hence, cannot meet
the requirements of the future high capacity ISTN. Frequency
reuse can be easily achieved in ground based cellular networks by limiting the coverage area of each cell within its
cell boundary in order to improve the spectral efficiency, but
this is not true for communications between the space and
terrestrial networks. Because the satellite is far away from the
ground, the coverage of a narrow communication beam will
be very large when it reaches the ground, resulting in very
low spectral efficiency. For the given available bandwidth,
low frequency reuse will significantly affect the overall capacity of the ISTN.
The capacity of the aerial backbones is proportional to the
bandwidth. The available bandwidth is limited in lower bands

to a few MHz up to a few hundred MHz. One way to increase
the bandwidth is to use a higher frequency band (such as
mmWave or THz). However, then the signal propagation
is affected by atmospheric losses, and the communication
distance is greatly reduced. Another way to increase the
capacity is to utilize multiple antennas to perform spatial
multiplexing, but the size and weight limits the number of
antennas that can be deployed on a satellite.
Taking all the above ways into consideration, it is believed
that using an mmWave communication system is the best
choice for realizing the high-speed RF backbone with data
rate up to 100 Gbps. Due to the over 10 GHz bandwidth in the
mmWave band (such as the 71-76 and 81-86 GHz E-band),
it can meet the spectrum requirements of the system. The
increase in transmit power can be solved with a high power
amplifier combined with an antenna array. The atmospheric
attenuation is below 0.3 dB/km and the total atmospheric and
cumulus loss (assuming 50 km link distance) is estimated to
be only about 10 dB. Multi-channel transmission can also
be implemented using line-of-sight multiple input multiple
output technology (LoS-MIMO).
Although being superior to microwave systems, currently
existing mmWave links still cannot meet the requirements
of the high capacity ISTN due to the lack of advanced
technologies. For instance, in-band full duplex has not been
adopted due to the difficulty in cancelling self-interference
especially in mmWave frequencies, thus only reaching 50%
of the potential data rate. Further, although LoS-MIMO has
been considered to achieve spatial multiplexing, the rank of
the channel is limited since the propagation distance is large
and the antenna spacing is small. Finally, yet importantly,
to reach longer communication distance, sufficiently high
transmit power is necessary. Even with the most powerful 40
dBm solid-state power sources in mmWave bands, achieving
longer communication distance beyond 50 km is still very
difficult. This calls for the employment of massive, preferably conformal and reconfigurable, antenna arrays.
Other enabling technologies to realize high-speed and
low-cost ISTN include the modelling of the mmWave



28 VOLUME 4, 2021


N. Rajatheva _et al._ : Scoring the Terabit/s Goal: Broadband Connectivity in 6G


**FIGURE 20.** Illustration of the ISTN architecture and its three layers. Yellow line: free space optics link; green line: mmWave link. Blue line: other microwave link).



aeronautical transmission channels and the dynamic behaviours of spaceborne and airborne networks especially
when LEO satellites are involved, three-dimensional (SpaceAir-Ground) networking and optimization, and high-speed
communication protocol optimization.


_**F. INTEGRATED WIDEBAND BROADCAST NETWORKS**_

The demands on network capabilities continue to increase,
among other aspects, in terms of capacity, availability and
cost. These increasing demands could be challenging to
networks that only support unicast and, more precisely, when
there are more users that require simultaneous service than
the APs are able to separate by beamforming. When some of
those users request the same data at the same time, broadcast
and multicast are suitable transport mechanisms [233]. They
are great options for large-scale delivery, since they permit
the transmission of the same content to a vast number of

devices within the covered area simultaneously, and with a
predefined quality-of-service.
Future 6G delivery networks need to be as flexible as
possible to respond to the needs of service providers, network
operators and users. Hence, broadcast and multicast mechanisms could be integrated into 6G networks as a flexible and
dynamic delivery option to enable a cost-efficient and scalable delivery of content and services in situations where the
APs have limited beamforming capabilities. More precisely,
if the AP can only send wide beams, then it should be able to
broadcast information over the coverage area of that beam.

6G also represents a great opportunity for the convergence



of mobile broadband and traditional broadcast networks,
usually used for TV video broadcasting [234]. Low-Power
Low-Tower cellular networks with adaptive beamforming
capabilities would benefit from the complementary coverage
provided by High-Power High-Tower broadcast networks
with fixed antenna patterns [235], as shown in Fig. 21. A
technology flexible enough to efficiently distribute content
over any of these networks would allow the 6G infrastructure
to better match the needs of future consumers and make
more efficient use of the existing tower infrastructure. The
convergence could be addressed from different perspectives,
i.e., the design of a single and highly efficient radio physical
layer; the use of common transport protocols across fixed and
mobile networks including broadcast, multicast and unicast
services; or a holistic approach that allows client applications
running on handsets to better understand and, therefore, adapt
to the capabilities of the underlying networks.


In Europe, it was decided to “ensure availability at least
until 2030 of the 470-694 MHz (‘sub-700 MHz’) frequency
band for the terrestrial provision of broadcasting services”

[236]. Some assignations are already happening in the USA,
where the 600 MHz frequency band is being assigned to
mobile broadband services [237]. The timing is very well
aligned with the release of 6G, and it is now when a full
convergence could take place.


A potential solution would be the use of a 6G wideband
broadcasting system, proposed in [238], [239]. By using 6G
wideband, all RF channels within a particular frequency band
could be used on all high-power high-tower transmitter sites



VOLUME 4, 2021 29


**FIGURE 21.** Network convergence and different topologies involved.


**FIGURE 22.** 6G wideband concept and comparison with traditional digital
broadcast (Source: [238]).


(i.e., frequency reuse-1). This is drastically different from
current broadcasting networks, where usually a frequency
reuse-7 is used to avoid inter-cell interference among transmitters. The entire wideband signal requires only half the
transmission power of a single traditional digital RF channel,
as shown in Fig. 22. Another advantage is that a similar
capacity could be obtained, thanks to the use of a much more
robust modulation and coding rate combination, since the
whole frequency band is employed. In terms of power, 6G
wideband also permits to transmit about 17 dB less power
(around 50 times) per RF channel (considering a bandwidth
of 8 MHz, typical from digital terrestrial broadcasting channels), although using more RF channels per station. This
leads to a total transmit power saving of around 90% [238].
Thanks to this higher spectrum use, the approach allows not
only for a dramatic reduction in fundamental power and cost,
but also about a 37-60% increase in capacity for the same
coverage as with current services.


**V. ENABLERS AT THE PROTOCOL/ALGORITHMIC**

**LEVEL**
The third factor in (1) is the spectral efficiency, which is
closely related to the modulation and coding, interference
management, and resource management in general. These are
some of the enabling technologies covered in this section.


_**A. CODING, MODULATION, WAVEFORM, AND DUPLEX**_



N. Rajatheva _et al._ : Scoring the Terabit/s Goal: Broadband Connectivity in 6G


1) Channel Coding
From 4G to 5G, the peak data rate has increased by 10100 times, and this trend is likely to continue with 6G. The
throughput of a single decoder in a 6G device will reach hundreds of Gbps. Infrastructure links are even more demanding
since they aggregate user throughput in a given cell or virtual
cell, which is expected to increase due to spatial multiplexing. However, it will be difficult to achieve such a high
throughput, only relying on the progress of integrated circuit
manufacturing technology within ten years. Solutions must
be found on the algorithm side as well. Both FEC code design
and corresponding encoding/decoding algorithms need to be
taken into account to reduce the decoding iterations and improve the decoder’s level of parallelism. Moreover, it is vital
for the decoder to achieve reasonably high energy efficiency.
To maintain the same energy consumption as in current
devices, the energy consumption per bit needs to be reduced
by 1-2 orders of magnitude. Implementation considerations
such as area efficiency (in Gbps/mm [2] ), energy efficiency
(in Tb/J), and absolute power consumption (W) place huge
challenges on FEC code design, decoder architecture, and
implementation [240].
FEC code design implies trade-offs between communications performance and high throughput. High communications performance typically requires complex decoding
algorithms and large number of iterations to achieve near
maximum-likelihood performance and large block lengths
to approach the Shannon bound. On the other hand, 6G
communication systems require flexibility in the codeword
length and coding rate. The most commonly used coding
schemes today are turbo codes, polar codes, and LDPC.
Their performance and throughput have already been pushed
towards the limit using 16 nm and 7 nm technology [240].
Trade-offs must be made between parallelization, pipelining,
iterations, and unrolling, linking with the code design and
decoder architecture. Future performance and efficiency improvements could come from CMOS scaling, but the picture
is quite complex [241]. Indeed, trade-offs must be made to
cope with issues such as power density/dark silicon, interconnect delays, variability, and reliability. Cost considerations
render the picture even more complex: costs due to silicon
area, design effort, test, yield and masks, explode at 7 nm
and below.

The channel coding scheme used in 6G high-reliability
scenarios must provide a lower error floor and better “waterfall” performance than that in 5G. Short and moderate length
codes with excellent performance need to be considered.
Polar codes, due to their error correction capabilities and
lack of error floor, might be the preferred choice in 6G.
However, state-of-the art CA-SCL decoding does not scale
up well with throughput due to the serial nature of the
algorithm. As a result, iterative algorithms like BP which
are more parallelizable have become a prime candidate for
channel decoding in high throughput data transmissions in
6G. However, as of now for polar codes, there exists a
significant performance gap between state-of-the art CA


30 VOLUME 4, 2021


N. Rajatheva _et al._ : Scoring the Terabit/s Goal: Broadband Connectivity in 6G


SCL decoding and BP. Hence, effort has been made towards
improving the performance of iterative algorithms. In [242],
authors propose a novel variant of BP called multi-trellis
belief propagation which is based on permuted factor graphs
of the polar code. Further improvements to the algorithm
are investigated in [243], [244]. In [245], along with a discussion on implication of various aspects of the polar code
factor graph on the performance, a new variant of the multitrellis BP decoder is proposed based on permutation of a
subgraph of the original factor graph (Fig. 23). This enables
the decoder to retain information about variable nodes in the

subgraphs, which are not permuted, reducing the required
number of iterations needed in-between the permutations.
Even though the iterative algorithms are prime candidates
in high throughput applications, progress has been made
on reducing the latency and improving the parallelizability
of algorithms based on successive cancellation. In [246]
authors propose methods to calculate the information bits
without traversing the complete binary tree on sub trees with
specific information bit and frozen bit patterns resulting in
latency improvements. Same idea is extended to successive
cancellation list decoding in [247]. Work in [248] proposes
some new algorithms to process new types of node patterns
that appear within multiple levels of pruned sub-trees, and it
enables the processing of certain nodes in parallel improving
the throughput of the decoder. Furthermore, modified polar
code constructions can be adopted to improve the performance of belief propagation by selecting the information bits
such that the minimum stopping set is larger [249]. In [250]
similar approach is adopted to improve the latency in SC
decoding. Here, the polar code construction is formulated
as an optimization problem which minimizes the complexity
under a mutual information based performance constraint.
Another potential avenue of improving the throughput is the
use of deep neural networks to approximate the iterative decoders which is called as deep unfolding [251], [252]. These
approaches could lead to better communication performance
but they will require significant advances in understanding
the behavior, robustness, and generalization of neural networks.


2) Modulation and Waveform
Modulation is another aspect that can be revised in 6G.
High-order QAM constellations have been used to improve
spectral efficiency in high SNR situations. However, because
of the non-linearity of hardware, the benefits obtained in
higher-order QAM constellations are gradually disappearing.
Non-uniform constellations have been adopted in the ATSC
3.0 standard for terrestrial broadcasting [253]. Probabilistic
shaping schemes as in [254], [255] provide in general even
better performance and are nowadays widely used for optical fiber communication, and they can be employed for
wireless backhaul channels [256]. Significant shaping gains
can already be achieved by simple modifications of the 5G
NR polar coding chain [257]–[259]. Similar approaches may
be used to generate transmit symbol sequences with arbi

#### **_x 1_** **_x 2_** **_x_**

_**3**_

#### **_x_**

_**4**_

#### **_x_**

_**5**_























_**6**_

#### **_x_**

_**7**_

#### **_x_**

_**8**_
_**x**_ _**8**_ _**x**_ _**8**_








|Col1|Col2|Col3|
|---|---|---|
||||
||||
||||



**FIGURE 23.** Partially permuted factor graph for polar code of block length 8.


trary probability distributions optimized for different channel
models (see e.g. [260]–[262] and the references therein).
Reducing the PAPR is another important technology direction in order to enable IoT with low-cost devices, edge coverage in THz communications, industrial-IoT applications with
high reliability, etc. There will be many different types of
demanding use cases in 6G, each with its own requirements.
No single waveform solution will address the requirements
of all scenarios. For example, as discussed in section III-A1,
the high-frequency scenario is faced with challenges such
as higher phase noise, larger propagation losses, and lower
power amplifier efficiency. Single-carrier waveforms might
be preferable over conventional multi-carrier waveforms to
overcome these challenges [263]. For indoor hotspots, the
requirements instead involve higher data rates and the need
for flexible user scheduling. Waveforms based on OFDM or
its variants exhibiting lower out-of-band emissions [264]–

[268] will remain a good option for this scenario. 6G needs a
high level of reconfigurability to become optimized towards
different use cases at different times or frequencies.
As previously mentioned, a critical impact from RF impairments is expected for THz systems. The main impairments include: IQ imbalance; oscillator phase noise;
and more generally non-linearities impacting amplitude and
phase. The non-linearities of analog RF front-end create
new challenges in both the modeling of circuits and the
design of mitigation techniques. It motivates the definition
for new modulation shaping, inherently robust to these RF
impairments. As an example, we consider the robustness to
the phase noise. According to [269], when the corner frequency of the oscillator is small in comparison to the system



VOLUME 4, 2021 31


bandwidth, the phase noise can be accurately modeled by
an uncorrelated Gaussian process. Using this mathematically
convenient assumption, it is demonstrated in [270] that using
a constellation defined upon a lattice in the amplitude-phase
domain is particularly relevant for phase noise channels.
More specifically, a constellation defined upon a lattice in the
amplitude-phase domain is robust to PN and leads to a lowcomplexity implementation. The Polar-QAM (PQAM) constellation is introduced in [270]. This constellation defines a
set of _M_ complex points placed on Γ _∈{_ 1 _,_ 2 _,_ 4 _, . . ., M_ _}_
concentric circles, i.e. amplitude levels. Each of the Γ circles
contains _M/_ Γ signal points. Any PQAM constellation are
hence entirely defined by two parameters, the modulation
order _M_ and the modulation shape Γ.
We therefore use the notation _M_ -PQAM(Γ). Examples of
different 16-PQAM(Γ) constellations are depicted in Fig. 24.
Some particular cases of the PQAM fall into well known
modulations: a _M_ -PQAM( _M/_ 2) describes an amplitudeshift keying while a _M_ -PQAM(1) is a phase-shift keying.
The PQAM is a structured definition of an APSK constellation. Combining this modulation scheme with a dedicated
demodulation – see for instance the polar metric detector
proposed in [270] – leads to a simple and robust transmission
technique for phase noise channels. With regard to RF poweramplifiers, directly related to the energy consumption of
transmitters, the PAPR is a key performance indicator for any
communication system. In the case of the PQAM, the PAPR
is given by

3 _·_ [2Γ] _[ −]_ [1] (12)

2Γ + 1 _[.]_


It can be noticed that the PAPR is an increasing function
of Γ and does not depend on the modulation order _M_ . We
see here the trade-off to be found between the PAPR and

the robustness to phase noise. Put it differently, increasing
Γ improves the robustness to phase noise, yet it increases the
PAPR in the same time. To fully understand the trade-offs
we are dealing with, let us present the results of numerical
simulations for the Gaussian phase noise channel, with two
phase noise variances, medium and strong. We consider
different settings of the PQAM combined with a state-of-theart channel coding scheme. The coding scheme is based on
the 5G-NR LDPC. It is implemented with an input packet
size of 1500 bytes and a coding rate ranging from 0 _._ 3 to 0 _._ 9.
A Packet Error Rate (PER) of 10 _[−]_ [2] is targeted for numerical
evaluations. The phase noise variances are _σ_ _φ_ [2] [= 10] _[−]_ [2] [ rad] [2]

for the medium phase noise case and _σ_ _φ_ [2] [= 10] _[−]_ [1] [ rad] [2] [ for]
the strong phase noise one. It should be emphasized that
the _σ_ _φ_ [2] [= 10] _[−]_ [1] [ rad] [2] [ (resp.] _[ σ]_ _φ_ [2] [= 10] _[−]_ [2] [ rad] [2] [) corresponds to]
an oscillator spectral density floor of _−_ 100 dBc/Hz (resp.
_−_ 110 dBc/Hz) for a bandwidth of 1 GHz [269]. Results
are depicted in Fig. 25. We also put for comparison the
performance of a QAM modulation with both the standard
Euclidean detector and an optimized joint amplitude-phase
detector, known to improve the demodulation performance
on phase noise channels [271].



N. Rajatheva _et al._ : Scoring the Terabit/s Goal: Broadband Connectivity in 6G


In the case of medium phase noise, the proposed PolarQAM scheme demonstrates a minor performance loss with
respect to the QAM. However, if the PAPR constraint becomes critical, the use of a Polar-QAM constellation with
a low PAPR can be envisaged. The performance in the
case of strong phase noise are particularly interesting to
analyze. First of all, we clearly observe that the achievable
information rate is severely limited when the QAM is used
due to phase noise. Conversely, the PQAM allows to reach
larger spectral-efficiencies. In the low SNR regime, it is
possible to considered low PAPR modulation, i.e. most of
the information bits are carried by the phase component. In
this regime the degradation from thermal noise is dominant
in comparison to phase noise. On the contrary in the high
SNR regime, performance are limited by phase noise. It is
clear from Fig. 25 that it is not possible to reach high rate
with low PAPR in this case. On the contrary to the QAM, the
Polar-QAM provides an additional degree of freedom in the
choice of the modulation scheme to find the best trade-off
between the level of PAPR and the phase noise robustness.
For this reason, the Polar-QAM offers an interesting modulation scheme for future coherent THz communication systems
targeting high spectral-efficiency.


3) Full-Duplex
The current wireless systems (e.g., 4G, 5G, WiFi) use TDD
or FDD methods, and the so-called half-duplex, where transmission and reception are not performed at the same time or
frequency. In contrast, the full-duplex or the IBFD technology allows a device to transmit and receive simultaneously
in the same frequency band. Full-duplex technology has the
potential to double the spectral efficiency and significantly
increase the throughput of wireless communication systems
and networks. The biggest hurdle in the implementation of
IBFD technology is self-interference, i.e., the interference
generated by the transmit signal to the received signal, which
can typically be over 100 dB higher than the receiver noise
floor. Three classes of cancellation techniques are usually
used to cope with self-interference: passive suppression,
analog cancellation, and digital cancellation (see e.g. [272]
and the references therein). Passive suppression involves
achieving high isolation between the transmit and receive
antennas before analog or digital cancellation. Analog cancellation in the RF domain is compulsory to avoid saturating
the receiving chain, and nonlinear digital cancellation in the
baseband is needed to further suppress self-interference, for
example, down to the level of the noise floor. The analog RF
cancellation can also be performed jointly with the digital
cancellation using a common processor.
The full-duplex technique has a wide range of benefits,
e.g., for relaying, bidirectional communication, cooperative
transmission in heterogeneous networks, and cognitive radio applications. Other prospective use cases include, for
instance, SWIPT, see e.g., [273]–[277] and the references
therein. The feasibility of full-duplex transmission has been
experimentally demonstrated in small-scale wireless commu


32 VOLUME 4, 2021


N. Rajatheva _et al._ : Scoring the Terabit/s Goal: Broadband Connectivity in 6G


2



|Col1|Col2|Col3|Col4|Col5|Col6|Col7|1<br>1|6 P-QA<br>6 P-QA|M(1)<br>M(2)|
|---|---|---|---|---|---|---|---|---|---|
||||||||1<br>|6 P-QA<br>|M(4)<br>|
|||||||1|1|6 P-QA<br>6 P-QA|M(8)<br>M(16)|
|||||||||||
|||||||||||
|||||||||||
|||||||||||
|||||||||||
|||||||||||
|||||||||||


0 0.2 0.4 0.6 0.8 1 1.2 1.4 1.6 1.8

Amplitude



1.5


1


0.5


0


-0.5


-1


-1.5


-2




|Col1|Col2|Col3|Col4|Col5|Col6|Col7|16 P-Q<br>16 P-Q|AM(1)<br>AM(2)|
|---|---|---|---|---|---|---|---|---|
||||||||16 P-Q<br>16 P-Q<br>16 P-Q|AM(4)<br>AM(8)<br>M(16)|
||||||||||
||||||||||
||||||||||
||||||||||
||||||||||
||||||||||
||||||||||



-2 -1.5 -1 -0.5 0 0.5 1 1.5 2

Real



1


0.8


0.6


0.4


0.2


0


-0.2


-0.4


-0.6


-0.8


-1



(a) (b)


**FIGURE 24.** Illustration of 16-PQAM(Γ) constellations in (a) the IQ plane and (b) the amplitude-phase domain.



nications environments, and it was also considered as an enabling technique for 5G but not yet adopted by 3GPP [278].
However, for the full-duplex technique to be successfully
employed in 6G wireless systems, there exist challenges on
all layers, ranging from the antenna and circuit design (e.g.,
due to hardware imperfection and nonlinearity, the non-ideal
frequency response of the circuits, phase noise, etc.), to the
development of theoretical foundations for wireless networks
with IBFD terminals. Note that IBFD becomes particularly
challenging when MIMO is taken into account and is even
more so with massive MIMO. Nevertheless, more antennas
brings also larger degrees of freedom. Some of the antennas
may be used for communication, while other may function
as support antennas to cancel the self-interference, or all
antennas used for both tasks jointly. The suitability of IBFD
technology for 6G is an open research area, where an interdisciplinary approach will be essential to meet the numerous
challenges ahead [272].


4) Protocol-Level Interference Management

Beamforming and sectorization have been effectively used
in 4G to manage the co-user interference. In 5G, massive
MIMO has been the main drive to control co-user inter
ference together with highly adaptive beamforming, which
helps to serve the spatially separated users non-orthogonally
by spatial multiplexing. However, there may be situations in
future communication systems when the interference management provided by massive MIMO is insufficient. For
example, if the arrays are far from the users, then the spatial
resolution of the antenna panels may be limited [279]. In
such cases, interference can be managed alternatively at the
protocol level. NOMA [280] and RS [281] schemes may be
two appropriate candidates for this purpose.
The NOMA techniques are mainly based on two domains:
power and code [282]. Power-domain NOMA refers to the
superposition of multiple messages using different transmit



powers so that users with higher SNRs can decode interfering
signals before decoding their own signal, while users with
lower SNRs can treat interference as noise [283]. Codedomain NOMA refers to the use of non-orthogonal spreading
codes, which provide users with higher SNRs after despreading, at the expense of additional interference [284]. Both
power-domain and code-domain NOMA have their advantages and disadvantages in terms of performance and implementation complexity when compared to each other. On
the other hand, RS is based on dividing the users’ messages
into private and common parts, where each user decodes its
private part and all users decode the common parts to extract
the data. From the implementation point of view, RS can be
viewed as a generalization of power-domain NOMA [285].

As for the interference management, while massive MIMO
treats any multi-user interference as noise, NOMA-based superposition coding with successive interference cancellation
fully decodes and removes the multi-user interference. On the
other hand, RS partially decodes the multi-user interference
and partially treats it as noise providing a trade-off between
fully decoding the interference and fully treating it as noise.
With their modified transceiver structures, both NOMA and
RS schemes may have increased complexity due to coding
and receiver processing. However, there may be prospective
6G use cases where interference management at the protocol
level could potentially provide sufficiently large gains to
outweigh the increased implementation complexity.

Some considerations of NOMA and RS for interference

management can be summarized as follows. In massive
connectivity scenarios, where many devices transmit small
packages intermittently, grant-free access using code-domain
NOMA or similar protocols could be very competitive. In
mmWave or THz communications, where the controllability
of the beamforming is limited by hardware constraints (e.g.,
phased arrays), NOMA and RS could enable multiple users to
share the same beam [286]. In VLC, where coherent adaptive



VOLUME 4, 2021 33


8


7


6


5


4


3


2


1





0

|Col1|Col2|Col3|Col4|Col5|Col6|
|---|---|---|---|---|---|
|Q<br>Q|AM modulatio<br>AM modulatio|n - standa<br>n - optim|rd detecto<br>zed detec|r<br>or||
|P<br>P<br>P|PR 3 dB<br>APR 4 dB<br>APR 5 dB|||||
|P<br>P<br>|APR 1 dB<br>APR 2 dB<br>|||||
|P|PR 0 dB|||||
|||||||
|||||||
|||||||

_−_ 5 0 5 10 15 20 25 30


SNR [dB]


8



7


6


5


4





3


2


1


0

|Col1|Col2|Col3|Col4|Col5|Col6|
|---|---|---|---|---|---|
|Q<br>Q|AM modulatio<br>AM modulatio|n - standa<br>n - optim|rd detecto<br>zed detec|r<br>or||
|P<br>P<br>P|PR 3 dB<br>APR 4 dB<br>APR 5 dB|||||
|P<br>P<br>|APR 1 dB<br>APR 2 dB<br>|||||
|P|PR 0 dB|||||
|||||||
|||||||
|||||||

_−_ 5 0 5 10 15 20 25 30


SNR [dB]


**FIGURE 25.** Achievable rate in bit/symbol of the QAM and PQAM modulation
schemes with a LDPC code for two phase noise configurations, medium and
strong. PAPR of the constellation is also depicted through a color map.


beamforming may be practically impossible for multiple access, NOMA and RS could be an appropriate solution [287].
For massive connectivity and achieving high data rates in 6G,
different implementations of NOMA techniques for cellular
networks could also be considered to increase the system
capacity [288]. In addition to the implementation of NOMA
and RS for interference management purposes, NOMAbased research that covers proposing simplified transceiver
structures and hybrid multiple access schemes could help the
improvement of interference management using NOMA and
RS.

On the other hand, there are several practical challenges
with implementing robust interference management at the
protocol level. One involves the error propagation effects that
appear when applying superposition coding and successive
interference cancellation to finite-sized data blocks [289].
Another issue is the high complexity in jointly selecting the
coding rates of all messages, to enable successful decoding
wherever needed, and conveying this information to the
users. Implementing adaptive modulation and coding is non


N. Rajatheva _et al._ : Scoring the Terabit/s Goal: Broadband Connectivity in 6G


trivial in fading environments with time-varying interference
and beamforming, even if it is carried out on a per-user basis.
With NOMA and RS, the selection of modulation/coding
becomes coupled between the users, making it an extremely
challenging resource allocation problem. Scheduling and
ARQ retransmissions are other tasks that become more complex when the user data transmission is coupled. Hence,
protocol-level interference management schemes need to be
investigated in detail under practical conditions in order to
determine suitable implementations for 6G use cases.


5) THz-Band Spatial Tuning

Besides combating the high propagation and absorption
losses at high frequencies (as previously highlighted in
Sec. IV), achieving any form of spatial multiplexing with
ultra-massive antenna configurations is hindered by the high
spatial channel correlation, caused by having a few angular
propagation paths. Even though the research community is
familiar with the impact of spatial correlation on MIMO
links, the severity of this phenomenon at mmWave/THz
frequencies and the peculiarities of THz devices result in
more severe challenges and novel opportunities, respectively.
This is particularly the case in doubly-massive MIMO configurations under LoS quasi-optical THz propagation, where
channels tend to be of very low rank. In such scenarios, it
is known that the physical separation between the antenna
elements at both the transmitter and the receiver can be

tuned to retain high channel ranks [290], [291]. In particular, for a given communication range _D_, there exists an
optimal separation ∆ between the antennas that guarantees
a sufficient number of eigenchannels to spatially multiplex
multiple data streams. With highly configurable THz antenna
architectures, especially plasmonic antennas, such optimal
separation can be tuned in real-time [95], [292]. However,
if _D_ is larger than the achievable Rayleigh distance, which
is a function of the physical array size, such tuning fails,
and ill-conditioned channels become unavoidable. Hence, the
Rayleigh distance serves as an important metric to capture
THz system performance. For a number of transmit antennas
_N_ _t_ and receive antennas _M_ _r_, this distance is expressed as

[293]
_D_ Ray = max _{M_ _r_ _, N_ _t_ _}_ ∆ _r_ ∆ _t_ _/λ,_ (13)


where _λ_ is the operating wavelength, and ∆ _r_ and ∆ _t_ are
the uniform separation between antennas at the receiver
and transmitter, respectively. Note that in a hybrid array-ofsubarrays configuration where multiplexing is conducted at
the level of subarrays, ∆ _r_ and ∆ _t_ can denote the corresponding separations between subarrays.
The achievable Rayleigh distances under different THz
LoS configurations are illustrated in Fig. 26, as a function
of antenna separations, array dimensions, and operating frequencies. A uniform planar array-of-subarrays configuration
is assumed, and two special cases of _M_ _r_ = _N_ _t_ = 128 _×_ 128
and _M_ _r_ = _N_ _t_ = 2 _×_ 2 are simulated, assuming ∆ _r_ = ∆ _t_ .
For antenna separations of a few millimeters, a large number



34 VOLUME 4, 2021


N. Rajatheva _et al._ : Scoring the Terabit/s Goal: Broadband Connectivity in 6G


10 [4]


10 [2]


10 [0]



10 [0]


10 [-1]











10 [-2]




|128 x128<br>elements|antenna|Col3|Col4|Col5|
|---|---|---|---|---|
||2 x 2 ant<br>elements|enna||1 THz<br>  0.8 THz<br>  0.6 THz<br>  0.4 THz<br>  0.2 THz<br>  0.1 THz|



0 20 40 60 80 100

Antenna Separation [mm]


**FIGURE 26.** Achievable Rayleigh distances under different THz LoS
configurations.


of antennas is required to realize multiplexing-achieving
distances beyond a few meters. Although for the same ∆,
larger arrays and higher frequencies result in larger Rayleigh
distances, for the same footprint, larger antenna numbers
incur a quadratic Rayleigh distance reduction in ∆. While
tens of mmWave-operating antennas typically occupy a few
square centimeters, a huge number of THz-operating antenna
elements can be embedded in few square millimeters. This
adaptability in design, when combined with numerical optimization, is what we call _spatial tuning_ . In particular, antenna
separations in plasmonic arrays can be reduced significantly,
to values much below _λ/_ 2, while still evading the effects of
mutual coupling. In such scenarios, to maintain an optimal ∆
for multiplexing, a large number of antennas can be kept idle.
Alternatively, spatial modulation [95] or multicarrier [294]
configurations can be applied. More importantly, antenna
oversampling can be further utilized to lower the spatiotemporal frequency-domain region of support of plane waves

[295], and this can be exploited for noise shaping techniques

[296]. While this discussion on channel characteristics is
illustrative, several assumptions need to be relaxed for the
exact characterization of THz channels, mainly by accounting for non-uniform array architectures [297], spherical wave
propagation [298], and wideband channels.
Efficient spatial tuning requires a sufficiently large array of
antennas (a uniform graphene sheet, for example), in which
antenna elements can be contiguously placed over a threedimensinoal structure, and subarrays can be virtually formed
and adapted. For a given target communication range, a
specific number of antenna elements is allotted in the corresponding subarrays to achieve the required beamforming
gain. Afterward, the diversity and multiplexing gains are
dictated by the possible utilizations of the subarrays, limited
by the number of available RF chains and the overall array
dimension. Note that frequency-interleaved multicarrier utilization of antenna arrays is feasible because each antenna
element can be tuned to target resonant frequency without



10 [-2]

20 40 60 80 100 120 140

SNR-dB


**FIGURE 27.** BER performance of spatial multiplexing: _f_ =1 THz, _D_ =1 m,
16 _×_ 16 subarrays, and 16-QAM.


modifying its physical dimensions. The latter can be attained
via electrostatic bias or material doping at high frequencies,
whereas software-defined plasmonic metamaterials can alternatively be used for sub-THz frequencies [93].
Spatial tuning can further be extended to IRS-assisted
THz NLoS environments. By controlling which element to
reflect, a spatial degree of freedom is added at the IRS level,
which could enhance the multiplexing gain of the NLoS
system, but at the expense of a reduced total reflected power.
Global solutions can be derived by jointly optimizing ∆ 1,
∆ 2, and ∆ 3, at the transmitting array, intermediate IRS,
and receiving array, respectively. Nevertheless, true optimizations allow arbitrary precoding and combining configurations
and arbitrary responses on all reflecting elements; the latter
bounds the performance of lower-complexity spatial tuning
techniques. Fig. 27 illustrates the importance of spatial
tuning in allocating antenna elements for beamforming and
maintaining channel orthogonality in a LoS scenario under
perfect alignment; by allocating 1000 antenna elements per
subarray, a 60 dB performance enhancement is noted and by
maintaining channel orthogonality error floors are avoided.


_**B. MACHINE LEARNING-AIDED ALGORITHMS**_

Wireless technology becomes more complicated for every
generation with so many sophisticated interconnected components to design and optimize. In recent research, there
have been an increasing interest in utilizing ML techniques to
complement the traditional model-driven algorithmic design
with data-driven approaches. There are two main motivations
for this [299]: modeling or algorithmic deficiencies.
The traditional model-driven approach is optimal when the
underlying models are accurate. This is often the case in the
physical layer, but not always. When the system operates
in complex propagation environments that have distinct but
unknown channel properties, there is a modeling deficiency
and, therefore, a potential benefit in tweaking the physical
layer using ML.
There are also situations where the data-driven approach



VOLUME 4, 2021 35


leads to highly complex algorithms, both when it comes to
computations, run-time, and the acquisition of the necessary
side information. This is called an algorithmic deficiency and
it can potentially be overcome by ML techniques, which can
learn how to take shortcuts in the algorithmic design. This is
particularly important when we need to have effective signal
processing for latency-critical applications and when jointly
optimizing many blocks in a communication system.
This section provides some concrete examples and potential future research directions.


1) End-to-End Learning
End-to-end learning of communication systems using ML
allows joint optimization of the transmitter and receiver
components in a single process instead of having the artificial
block structure as in conventional communication systems. In

[300], autoencoder concept is used to learn the system minimizing the end-to-end message reconstruction error. Extensions of the autoencoder implementation for different system
models and channel conditions can be found in [301]–[304].
Furthermore, different approaches for end-to-end learning
have been considered when the channel model is unknown
or difficult to model analytically [305]–[307]. Specifically,

[307] provides an iterative algorithm for the training of
communication systems with an unknown channel model
or with non-differentiable components. A unified multi-task
deep neural network framework for NOMA is proposed in

[308], consisting of different components which utilize datadriven and communication-domain expertise approaches to
achieve end-to-end optimization.
An example scenario of end-to-end learning is discussed
below. Given the task of transmitting message _s_ out of
_M_ possible messages from the transmitter to receiver over
the AWGN channel using _n_ complex channel uses with a
minimum error, the system is modelled as an autoencoder
and implemented as a feedforward neural network. A model
similar to the model proposed in [300] is implemented with
slight variations [309]. The transmitter consists of multiple
dense layers followed by a normalization layer to guarantee
the physical constraints of the transmit signal. During the
model training, the AWGN channel is modelled as an additive noise layer with a fixed variance _β_ = (2 _RE_ _b_ _/N_ 0 ) _[−]_ [1],
where _E_ _b_ _/N_ 0 denotes the energy per bit ( _E_ _b_ ) to noise power
spectral density ( _N_ 0 ) ratio. The receiver also consists of multiple dense layers and an output layer with softmax activation
to output the highest probable estimated message.
The model is trained in end-to-end manner on the set of

all possible messages _s ∈_ M using the categorical crossentropy loss function. A training set of 1,000,000 randomly
generated messages with _E_ _b_ _/N_ 0 = 5 dB is used for model
training. The autoencoder learns optimum transmit symbols
after the model training and the BER performance of the
learnt transmit mechanism is evaluated with another set of
1,000,000 random messages across the 0 dB to 8 dB _E_ _b_ _/N_ 0
range. Fig. 28 and Fig. 29 show the BER performance of different autoencoder models in comparison with conventional



N. Rajatheva _et al._ : Scoring the Terabit/s Goal: Broadband Connectivity in 6G


BPSK and QPSK modulation schemes. It can be observed
that increasing the message size _M_ improves the BER performance of the autoencoder resulting in a better BER than
the conventional BPSK and QPSK schemes.
Practical aspects considering the model training implementations, methods to accompany unknown channel conditions and varying channel statistics in long-term, transmitterreceiver synchronization etc. are some of the potential future
research directions.


2) Joint Channel Estimation and Detection

Channel estimation, equalization, and signal detection are
three tasks in the physical layer that are relatively easy to
carry out individually. However, their optimal joint design
is substantially more complicated, making ML a potential
shortcut. In [310], a learning-based joint channel estimation and signal detection algorithm is proposed for OFDM
systems, which can reduce the signaling overhead and deal
with nonlinear clipping. In [311], joint channel estimation
and signal detection is carried out when both the transmitter
and receiver are subject to hardware impairments with an
unknown model. Online learning-based channel estimation
and equalization is proposed in [312], which can jointly
handle fading channels and nonlinear distortion.
Methods to overcome the challenge of offline model training, which causes performance degradation due to the discrepancies between the real channels and training channels,
need to be considered. ML implementations with online
training and constructing training data to match real-world
channel conditions are potential future directions in this
regard.


10 [0]


10 [-1]


10 [-2]


10 [-3]



10 [-6]

|Col1|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|
|---|---|---|---|---|---|---|---|---|
||||||||||
||||||||||
||||||||||
||||||||||
|||M=2, n=<br>M=4, n=|1 Autoe<br>2 Autoe|ncoder<br>coder|||||
|||M=8, n=<br>|3 Autoe<br>|ncoder<br>|||||
|||M=16, n<br>=256,|=4 Auto<br>=8 Aut|ncoder<br>encode|||||
|||PSK|||||||
||||||||||
||||||||||


0 1 2 3 4 5 6 7 8
E b /N 0 (dB)


**FIGURE 28.** BER performance of the _R_ = 1 bits/channel use systems
compared with theoretical AWGN BPSK performance.


3) Resource Management

Resource management problems are often combinatorial in
nature, which implies that optimal exhaustive-search based



10 [-4]


10 [-5]





36 VOLUME 4, 2021


N. Rajatheva _et al._ : Scoring the Terabit/s Goal: Broadband Connectivity in 6G


10 [0]


10 [-1]


10 [-2]


10 [-3]



10 [-4]




|Col1|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|Col10|Col11|
|---|---|---|---|---|---|---|---|---|---|---|
||||||||||||
||||||||||||
||||||||||||
||||||||||||
||||||||||||
||||||||||||
||||||||||||
||||||||||||
||||||||||||
||||||||||||
||||M=4, n=<br>|1 Autoe<br>|ncoder<br>||||||
||||||||||||
||||M=64, n<br>|=3 Auto<br>|encoder<br>||||||
||||M=256,<br>|n=4 Au|oencode|r|||||
||||||||||||
||||||||||||
||||||||||||



0 1 2 3 4 5 6 7 8
E b /N 0 (dB)


**FIGURE 29.** BER performance of the _R_ = 2 bits/channel use systems
compared with theoretical AWGN QPSK performance.


algorithms are impossible to utilize in practice. This provides
an opportunity for ML-based algorithms to outperform existing suboptimal approaches. A generic resource allocation
framework is provided in [313], whereas a joint beamforming, power control, and interference coordination approach
is proposed in [314]. Some examples of hybrid beamforming in mmWave MIMO systems [315]–[318] utilize MLbased hybrid precoder-combiner designs, resulting in close
performance and reduced complexity compared to the conventional exhaustive search-based optimization techniques.
The work in [319]–[322] utilize different supervised learning
and reinforcement learning techniques to learn and predict
the optimum power allocation in the network dynamically.
Some potential research areas which are already being
studied include power control, beamforming in massive
MIMO and in cell-free environments, predictive scheduling
and resource allocation, etc. Reinforcement learning frameworks and transfer learning techniques to learn, adapt, and
optimize for varying conditions over time are expected to
be useful for performing resource management tasks with
minimal supervision [323].


_**C. CODED CACHING**_

Coded caching is originally proposed by Maddah-Ali and
Niesen in [324], as a way to increase the data rate with the
help of cache memories available throughout the network. It
enables a global caching gain, proportional to the total cache
size of all users in the network, to be achieved in addition
to the local caching gain at each user. This additional gain
is achieved by multicasting carefully created codewords to
various user groups, so that each codeword contains useful
data for every user in the target group. Technically, if _K_ is
the user count, _M_ denotes the cache size at each user and
_N_ represents the number of files in the library, using CC the
required data size to be transmitted over the broadcast link



can be reduced by a factor of 1 + _t_ (equivalently, the data
rate can be increased by a factor of 1 + _t_ ), where _t_ = _[KM]_ _N_ is

called the CC gain.
Interestingly, the CC gain is not only achievable in multiantenna communications, but is also additive with the spatial
multiplexing gain of using multiple antennas [325]–[327]. In
fact, caching non-overlapping file fragments at user terminals provides multicasting opportunities, as multiple users
can be potentially served with a single multicast message.
Therefore, the additive gain of CC in multi-antenna communications can be achieved through multicast beamforming
of multiple parallel (partially overlapping) CC codewords to
larger sets of users, while removing (or suppressing) intercodeword interference with the help of carefully designed
beamforming vectors. The generalized multicast beamforming design enables innovative, flexible resource allocation
schemes for CC. Depending on the spatial degrees of freedom and the available power resources, a varying number of
multicast messages can be transmitted in parallel to distinct
subsets of users [328].
The multi-antenna CC structure also provides more flexibility in reducing subpacketization, defined as the number of
smaller files each file should be split into, for the CC structure
to work properly. An exponentially growing subpacketization
requirement is known to be a major problem in implementing
original single- and multi-antenna CC schemes [329]. However, recently it is shown that in multi-antenna CC, especially
when the spatial multiplexing gain is larger than the CC
gain, linear or near-linear subpacketization growth is possible through well-defined algorithms [329]–[331]. Overall,
the nice implementation possibility of CC in multi-antenna
setups makes it a desirable option to be implemented in future
wireless networks, where MIMO techniques are considered
to be a core part.
The number of multimedia applications benefiting from
CC is expected to grow in the future. One envisioned scenario
assumes an extended reality or hyper-reality environment
(e.g., educational, industrial, gaming, defense, social networking), as depicted in Fig. 30. A large group of users is
submerged in a network-based immersive application, which
runs on high-end eye-wear that requires heavy multimedia
traffic and is bound to guarantee a well-defined quality-ofexperience level for every user in the operating theatre. The
users are scattered across the area covered by the application
and can move freely, and their streamed data is unique and
highly location- and time-dependent. Notably, a large part of
the rich multimedia content for rendering a certain viewpoint
is common among the users. This offers the opportunity for
efficient use of pooled memory resources through intelligent
cache placement and multicast content delivery mechanisms.
In such a scenario, the possibility of caching on the user
devices and computation offloading onto the network edge
could potentially deliver high-throughput, low-latency traffic,
while ensuring its stability and reliability for a truly immersive experience. The fact that modern mobile devices are continuing to increase their storage capacity (which is one of the



VOLUME 4, 2021 37


N. Rajatheva _et al._ : Scoring the Terabit/s Goal: Broadband Connectivity in 6G


**FIGURE 30.** Immersive viewing scenario with coded caching.



cheapest network resources) makes CC especially beneficial
given the uniqueness of this use case, where the popularity
of limited and location-dependent content becomes much
higher than in any traditional network.


Alongside its strong features, there are many practical
issues with CC that must be addressed to enable its im
plementation in 6G. Most importantly, various parameters
affecting the performance and complexity of CC schemes
need to be identified, and possible trade-offs among them has
to be clarified. For example, one such trade-off is depicted
in Fig. 31 ( _L_ is the number of antennas at the transmitter side), in which the performance of the multi-antenna
CC scheme in [326], [328] is compared with the reducedsubpacketization scheme of [330], at various SNR levels and
for fully-optimized (MMSE-type) and zero-force beamforming strategies. In general, the MS performs better than the
RED; and optimized beamforming provides better results
than zero-forcing. However, the MS scheme requires exponentially growing subpacketization, making it impractical for
networks with even moderate number of users; and optimized
beamforming requires non-convex optimization problems to
be solved through computationally intensive methods such as
successive convex approximation [328]. On the other hand,
the difference between various strategies is dependent on
the SNR regime [332]. For example, in the figure it is clear
that when the SNR is below 15 dB, the RED scheme with
optimized beamforming outperforms MS strategy with zeroforcing; while for SNR above 15 dB this comparison is no
longer valid. Moreover, the gap between optimized and zeroforce beamformers vanishes at the high-SNR regime.

The provided example only clarifies the interaction between the subpacketization, beamforming strategy and op


erating SNR regime. It is shown that as the CC operation is
heavily dependent on the underlying multicasting implementation, its performance is degraded in case some users in the
network are suffering from poor channel conditions [333].
In addition, the energy efficiency of CC schemes is largely
unaddressed in the literature. In fact, CC schemes require
a proactive cache placement phase during which the cache
memories in the network are filled with data chunks from all
files in the library. This imposes a communication overhead
and necessitates excess energy consumption for the required
data transfer, which can considerably degrade the energy
efficiency of the underlying CC scheme.







|40<br>[nats/s]<br>30<br>Rate<br>20 Symmetric<br>10|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|Col10|Col11|Col12|Col13|Col14|Col15|Col16|Col17|Col18|Col19|Col20|Col21|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|10<br>20<br>30<br>40<br>Symmetric Rate [nats/s]|||S|, O|pti|i|e||||||||||||||
|10<br>20<br>30<br>40<br>Symmetric Rate [nats/s]|||||||||||||||||||||
|10<br>20<br>30<br>40<br>Symmetric Rate [nats/s]|||S|, Z|ero|fo|rc|e|||||||||||||
|10<br>20<br>30<br>40<br>Symmetric Rate [nats/s]|||||||||||||||||||||
|10<br>20<br>30<br>40<br>Symmetric Rate [nats/s]|||E|,|pti|m|z|d|||||||||||||
|10<br>20<br>30<br>40<br>Symmetric Rate [nats/s]|||E|,|er|-f|r|e|||||||||||||
|10<br>20<br>30<br>40<br>Symmetric Rate [nats/s]|||||||||||||||||||||
|10<br>20<br>30<br>40<br>Symmetric Rate [nats/s]|||||||||||||||||||||
|10<br>20<br>30<br>40<br>Symmetric Rate [nats/s]|||||||||||||||||||||
|10<br>20<br>30<br>40<br>Symmetric Rate [nats/s]|||||||||||||||||||||
|10<br>20<br>30<br>40<br>Symmetric Rate [nats/s]|||||||||||||||||||||
|10<br>20<br>30<br>40<br>Symmetric Rate [nats/s]|||||||||||||||||||||
|10<br>20<br>30<br>40<br>Symmetric Rate [nats/s]|||||||||||||||||||||
|10<br>20<br>30<br>40<br>Symmetric Rate [nats/s]|||||||||||||||||||||
|10<br>20<br>30<br>40<br>Symmetric Rate [nats/s]|||||||||||||||||||||
|10<br>20<br>30<br>40<br>Symmetric Rate [nats/s]|||||||||||||||||||||
|10<br>20<br>30<br>40<br>Symmetric Rate [nats/s]|||||||||||||||||||||
|10<br>20<br>30<br>40<br>Symmetric Rate [nats/s]|||||||||||||||||||||
|10<br>20<br>30<br>40<br>Symmetric Rate [nats/s]|||||||||||||||||||||
|10<br>20<br>30<br>40<br>Symmetric Rate [nats/s]|||||||||||||||||||||
|10<br>20<br>30<br>40<br>Symmetric Rate [nats/s]|||||||||||||||||||||
|10<br>20<br>30<br>40<br>Symmetric Rate [nats/s]|||||||||||||||||||||
|10<br>20<br>30<br>40<br>Symmetric Rate [nats/s]|||||||||||||||||||||
|10<br>20<br>30<br>40<br>Symmetric Rate [nats/s]|||||||||||||||||||||
|10<br>20<br>30<br>40<br>Symmetric Rate [nats/s]|||||||||||||||||||||
|10<br>20<br>30<br>40<br>Symmetric Rate [nats/s]|||||||||||||||||||||


0 10 20 30 40


SNR [dB]


**FIGURE 31.** Rate vs SNR, _K_ = 6, _t_ = 2, _L_ = 3.



38 VOLUME 4, 2021


N. Rajatheva _et al._ : Scoring the Terabit/s Goal: Broadband Connectivity in 6G


**VI. 6G BROADBAND CONNECTIVITY FOR**

**RURAL/UNDERDEVELOPED AREAS**

Around half of the world population, or almost 4 billion people, still lack broadband connectivity. Wireless connectivity
of some form is the only realistic alternative for most people
in large rural areas since building a new wired infrastructure
is typically cost prohibitive. While each new wireless generation so far has focused on giving higher data rates where
the population is most concentrated, they have offered much
less benefit in rural regions and sometimes resulted in even
worse coverage than the previous generations, which has
resulted in an increased digital divide. This unbalanced trend
still holds true today. Many emerging wireless technologies
are primarily targeted at improving the wireless connectivity
where it is already relatively good; at short distances to
the access network and to serve densely packed users and
devices with even higher capacity. While there may be some
improvement also in rural areas and to remote locations, the
gain is usually much lower than in urban areas.
Any technology that aims to reduce the digital divide needs
to offer as uniform performance as possible over as large
area as possible. One such technology is massive MIMO,
which can improve the performance at the cell edge considerably given an altruistic max-min power allocation. While
conventional high-gain sector antennas provide array gains in
the cell center, the highly adaptive beamforming of massive
MIMO can provide array gains wherever the user is located
in the coverage area [134, Sec. 6.1]. However, this comes at
the price of a reduced sum-rate which may be economically
unfeasible for an operator, since the long-distance users will
consume a large portion of the available power for an AP
in a cellular system, resulting in comparable much lower
rates for nearby users compared to uniform power allocation.
Long-range users may also have insufficient transmit power
in the uplink. Cell-free massive MIMO, however, with maxmin power control provide a much higher energy efficiency
and throughput per user in rural environments compared to a
single cell cellular system. This is achieved due to the lower
path-loss and that only a few antennas need to transmit on full
power in cell-free systems with distributed antennas [334].
Long-range massive MIMO can greatly increase the range
and coverage of an AP and increase the capacity also at long
distances, e.g., using very tall towers [335]. Coherent joint
transmission between adjacent APs can be utilized to enhance the performance and diversity at cell edges, effectively
creating a long-range cell-free massive MIMO system. However, challenges exist in that long-distance LOS dominated
channels are more sensitive to shadowing and blocking, and
frequent slow fading in general, which impose practical challenges for massive MIMO in rural areas. Unfortunate homes
and places may be in a permanent deep shadow and thus always out of coverage. The near-far problem is often enhanced
due to the nature of the wireless channels in rural areas,
with typically decreasing spatial diversity at longer distances.
Coherent joint transmissions from multiple distributed APs
in a cell-free arrangement require more backhaul/fronthaul



capacity and are subject to large delays. The synchronization
of widely distributed transmitters is also highly challenging
due to large variations in propagation delays.
LEO satellite constellations can offer almost uniform cov
erage and high capacity over huge rural areas, including seas
and oceans, and to other remote locations using mmWaves
and THz frequencies. They can also be used in conjunction
with HAPS. Since these solutions view the coverage area
high from above, they can fill in the coverage gaps of terrestrial long-range massive MIMO, as described above, and
also offer large-capacity backhaul links for remotely located
ground-based APs. High array gains are possible by using
many antenna elements on the satellites and HAPS. Each
user can either be served by the best available satellite/HAPS,
or by coherent transmission from multiple ones. The coordination can be enabled via optical inter-satellite connections.
Coherent transmission from satellites and ground-based APs
is also possible, to effectively create a space-based cellfree network. Many challenges exist, however, in that a
large number of LEO satellites are required to make them
constantly available in rural regions, which is associated with
a high deployment cost. Interference may arise between uncoordinated satellites and terrestrial systems using the same
radio spectrum. The large pathloss per antenna element requires large antenna arrays and adaptive beamforming, since
LEO satellites are constantly in motion. The transmit power
is limited, particularly in the uplink. Coherent transmission
from multiple satellites, or between space and the ground,
requires higher fronthaul capacity and varying delays cause
synchronization issues.
IRSs can be deployed on hills and mountains to remove
coverage holes in existing networks by reflecting signals
from APs or satellites towards places where the LoS path is
blocked and the natural multipath propagation is insufficient.
Fig. 10 illustrates such a case when the direct path is heavily
attenuated (Case 1) and where an IRS improves the spectral
efficiency considerably by reflecting many coherent signal
paths to the user. An IRS can be powered by solar panels
or other renewable sources. The cost is low compared to
deploying and operating additional APs since no backhaul
infrastructure or connection to the power grid is needed.
However, the larger the propagation distance becomes, the
larger the reflecting surface area needs to be in order to
counteract the large path-losses. Remote control of the IRS
is challenging and might only support fixed access or low
mobility. An IRS deployed in nature can be exposed to harsh
climate and weather conditions and a subject of sabotage.


**VII. CONCLUDING REMARKS**

This paper has provided a survey of the most promising
candidate technologies at the PHY and MAC layer for the
realization of Tbps wireless connectivity in future 6G wireless networks. These technologies, named enablers, have
been categorized into three separate classes, i.e. enablers
at the spectrum level, at the infrastructure level, and at the
protocol/algorithmic level.



VOLUME 4, 2021 39


N. Rajatheva _et al._ : Scoring the Terabit/s Goal: Broadband Connectivity in 6G


**TABLE 6.** Summary of Challenges, Potential 6G Solutions, and Open Research Questions












|Challenges|Potential 6G solutions|Open research questions|
|---|---|---|
|~~Stable service quality in coverage area~~<br>|~~User-centric cell-free massive MIMO~~<br>|~~Scalable synchronization, control, and resource~~<br>allocation<br>|
|~~Coverage improvements~~<br>|~~Integration of a spaceborne layer, ultra-massive~~<br>MIMO from tall towers, intelligent reﬂecting sur-<br>faces<br>|~~Joint control of space and ground based APs,~~<br>real-time control of IRS<br>|
|~~Extremely wide bandwidths~~<br>|~~Sub-THz, VLC~~<br>|~~Hardware development and mitigation of impair-~~<br>ments<br>|
|~~Reduced latency~~<br>|~~Faster forward error correcting schemes, wider~~<br>bandwidths<br>|~~Efﬁcient encoding and decoding algorithms~~<br>|
|~~Efﬁcient spectrum utilization~~<br>|~~Ultra-massive MIMO, waveform adaptation, in-~~<br>terference cancellation<br>|~~Holographic radio, use-case based waveforms,~~<br>full-duplex, rate-splitting<br>|
|~~Efﬁcient backhaul infrastructure~~<br>|~~Integrated access and backhauling~~<br>|~~Dynamic resource allocation framework using~~<br>space and frequency domains<br>|
|~~Smart radio environment~~<br>|~~Intelligent reﬂecting surfaces~~<br>|~~Channel estimation, hardware development, re-~~<br>mote control<br>|
|~~Energy efﬁciency~~<br>|~~Cell-free massive MIMO, suitable modulation~~<br>techniques<br>|~~Novel modulation methods with limited hard-~~<br>ware complexity<br>|
|~~Modeling or algorithmic deﬁciencies in complex~~<br>and dynamic scenarios|~~ML-/AI-based, model-free, data-driven learning~~<br>and optimization techniques|~~End-to-end learning/joint optimization, unsuper-~~<br>vised learning for radio resource management|



Our vision is that with 6G wireless networks there will be

a paradigm shift in the way users are supported, moving from
the network-centric view where networks are deployed to deliver extreme peak rates in special cases to a user-centric view
where consistently high rates are prioritized. Such ubiquitous
connectivity can be delivered by cell-free massive MIMO and
IAB, and complemented by IRSs. The sub-6 GHz spectrum
will continue defining the wide-area coverage and 1 Tbps
can be reached in this spectrum range by making use of infrastructure enablers for extremely high spatial multiplexing.
A significant effort is needed in sub-THz and THz bands
to achieve short-range connectivity with data rates in the
vicinity of 1 Tbps. The extremely wide bandwidths can take
us far towards the goal, but it challenging to maintain a decent
spectral efficiency. Novel coding, modulation and waveforms
will be needed to support extreme data rates with manageable
complexity and robustness to the hardware impairments that
will increase with the carrier frequency. In situations where
the beamforming capabilities offered by the ultra-massive
MIMO technology are insufficient to manage interference in
the spatial domain, coding methods based on rate-splitting
or broadcasting can be also utilized. The efficiency can also
be improved by making use of caching techniques. Seamless
integration between satellites and terrestrial networks will
ensure that large populations outside the urban centers will
be reached by high-quality broadband connectivity.

Throughout the paper the main challenges related to the
use of the described technologies have been highlighted
and discussed. For reader’s convenience, these are briefly
summarized in Table 6.

Our hope is that this paper will help accelerating the
interest of the scientific community towards these problems,
so that one day not so far into the future we will be able to
score the Terabit/s goal for broadband connectivity for everyone. It will be essential both for short-range communication
links and to handle the massive traffic from a large number



devices.


**REFERENCES**


[1] M. Latva-aho and K. Leppänen (eds.), “Key drivers and research
challenges for 6G ubiquitous wireless intelligence,” _White Paper_, Sept.
[2019. [Online]. Available: http://urn.fi/urn:isbn:9789526223544](http://urn.fi/urn:isbn:9789526223544)

[2] K. David and H. Berndt, “6G vision and requirements: Is there any need
for beyond 5G?” _IEEE Veh. Technol. Mag._, vol. 13, no. 3, pp. 72–80, Sep
2018.

[3] M. Giordani, M. Polese, M. Mezzavilla, S. Rangan, and M. Zorzi,
“Toward 6G networks: Use cases and technologies,” _IEEE Commun._
_Mag._, vol. 58, no. 3, pp. 55–61, Mar. 2020.

[4] Z. Zhang _et al._, “6G wireless networks: Vision, requirements, architecture, and key technologies,” _IEEE Veh. Technol. Mag._, vol. 14, no. 3, pp.
28–41, Sept. 2019.

[5] E. Calvanese Strinati, S. Barbarossa, J. L. Gonzalez-Jimenez, D. Ktenas,
N. Cassiau, L. Maret, and C. Dehos, “6G: The next frontier: From
holographic messaging to artificial intelligence using subterahertz and
visible light communication,” _IEEE Veh. Technol. Mag._, vol. 14, no. 3,
pp. 42–50, Sept. 2019.

[6] N. Rajatheva _et al._, “White paper on broadband connectivity in 6G,”
[June 2020. [Online]. Available: http://urn.fi/urn:isbn:9789526226798](http://urn.fi/urn:isbn:9789526226798)

[7] P. Yang, Y. Xiao, M. Xiao, and S. Li, “6G wireless communications:
Vision and potential techniques,” _IEEE Network_, vol. 33, no. 4, pp. 70–
75, Jul 2019.

[8] S. Dang, O. Amin, B. Shihada, and M.-S. Alouini, “What should 6G be?”
_Nature Electron._, vol. 3, no. 1, pp. 20–29, 2020.

[9] H. Viswanathan and P. E. Mogensen, “Communications in the 6G era,”
_IEEE Access_, vol. 8, pp. 57 063–57 074, 2020.

[10] G. Gui, M. Liu, F. Tang, N. Kato, and F. Adachi, “6G: Opening new
horizons for integration of comfort, security and intelligence,” _IEEE_
_Wireless Communications_, 2020.

[11] Networld2020, “Strategic research and innovation agenda
(SRIA) 2021–27: Smart networks in the context of NGI,”
May 2020. [Online]. Available: [https://www.networld2020.eu/](https://www.networld2020.eu/sria-public-consultation-smart-networks-in-the-context-of-ngi/)
[sria-public-consultation-smart-networks-in-the-context-of-ngi/](https://www.networld2020.eu/sria-public-consultation-smart-networks-in-the-context-of-ngi/)

[12] I. Akyildiz, A. Kak, and S. Nie, “6G and beyond: The future of wireless
communications systems,” _IEEE Access (under review)_, 2020.

[13] E. Björnson, L. Sanguinetti, H. Wymeersch, J. Hoydis, and T. L.
Marzetta, “Massive MIMO is a reality—What is next? Five promising
research directions for antenna arrays,” _Digit. Signal Process._, vol. 94,
pp. 3–20, Nov. 2019.

[14] 3GPP TS 38.104, _NR; Base Station (BS) radio transmission and recep-_
_tion_, v16.3.0 (2020-03), Release 16.

[15] 3GPP TS 38.211, _NR;Physical channels and modulation_, v16.1.0 (202003), Release 16.



40 VOLUME 4, 2021


N. Rajatheva _et al._ : Scoring the Terabit/s Goal: Broadband Connectivity in 6G


[16] 3GPP TS 38.300, _NR; NR and NG-RAN Overall Description; Stage 2_,
v16.1.0 (2020-03), Release 16.

[17] 3GPP TS 38.213, _NR; Physical layer procedures for control_, v16.1.0
(2020-03), Release 16.

[18] “Samsung, "New WID: Further Enhancements on MIMO for NR", RP193133, 3GPP TSG RAN Meeting #86,” Dec. 2019.

[19] 3GPP TS 38.874, _NR; Study on Integrated Access and Backhaul_, v16.0.0
(2018-12), Release 16.

[20] 3GPP RP-193248, _New Work Item on NR support of Multicast and_
_Broadcast Services_, Dec. 2019.

[21] J. Doré _et al._, “Technology roadmap for beyond 5G wireless connectivity
in D-band,” in _Proc. 6G Wireless Summit (6G SUMMIT)_, 2020.

[22] K. Guan _et al._, “5G channel models for railway use cases at mmwave
band and the path towards terahertz,” _IEEE Intell. Transp. Syst. Mag._, pp.
1–1, 2020.

[23] ITU-R M.2134, _Requirements related to technical performance for IMT-_
_Advanced radio interface(s)_, 2008.

[24] ITU-R M.2410, _Minimum requirements related to technical performance_
_for IMT-2020 radio interface(s)_, 2017.

[25] E. Björnson and E. G. Larsson, “How energy-efficient can a wireless
communication system become?” in _Proc. Asilomar Conf. Signals, Syst.,_
_and Comput. (ASILOMAR)_, 2018.

[26] N. H. Mahmood _et al._, “White paper on critical and massive machine
type communication towards 6G,” June 2020. [Online]. Available:
[http://urn.fi/urn:isbn:9789526226781](http://urn.fi/urn:isbn:9789526226781)

[27] A. Pouttu (ed.), “6G White paper on validation and trials for
verticals towards 2030’s,” _White Paper_, June 2020. [Online]. Available:
[http://urn.fi/urn:isbn:9789526226811](http://urn.fi/urn:isbn:9789526226811)

[28] “Intel Corporation, "New SID: Study on supporting NR from 52.6 GHz
to 71 GHz", RP-193259, 3GPP TSG RAN Meeting #86,” Dec. 2019.

[29] “FCC Fact Sheet, Spectrum Horizons,” February 2019. [Online].
[Available: https://docs.fcc.gov/public/attachments/DOC-356297A1.pdf](https://docs.fcc.gov/public/attachments/DOC-356297A1.pdf)

[30] I. F. Akyildiz, J. M. Jornet, and C. Han, “Teranets: Ultra-broadband communication networks in the Terahertz band,” _IEEE Wireless Commun._,
vol. 21, no. 4, pp. 130–135, Aug. 2014.

[31] T. Kürner and S. Priebe, “Towards THz communications-status in
research, standardization and regulation,” _J. Infrared Millimeter THz_
_Waves_, vol. 35, no. 1, pp. 53–62, Jan. 2014.

[32] K. Sengupta, T. Nagatsuma, and D. M. Mittleman, “Terahertz Integrated
Electronic and Hybrid Electronic–Photonic Systems,” _Nature Electron._,
vol. 1, no. 12, pp. 622–635, 2018.

[33] A. Nikpaik, A. H. M. Shirazi, A. Nabavi, S. Mirabbasi, and S. Shekhar,
“A 219-to-231 GHz Frequency-multiplier-based VCO with˜ 3% Peak
DC-to-RF Efficiency in 65-nm CMOS,” _IEEE Journal of Solid-State_
_Circuits_, vol. 53, no. 2, pp. 389–403, 2018.

[34] H. Aghasi, A. Cathelin, and E. Afshari, “A 0.92-THz SiGe Power
Radiator Based on a Nonlinear Theory for Harmonic Generation,” _IEEE_
_Journal of Solid-State Circuits_, vol. 52, no. 2, pp. 406–422, 2017.

[35] W. R. Deal _et al._, “A 666 GHz Demonstration Crosslink with 9.5 Gbps
Data Rate,” in _2017 IEEE MTT-S International Microwave Symposium_
_(IMS)_ . IEEE, 2017, pp. 233–235.

[36] I. Mehdi, J. V. Siles, C. Lee, and E. Schlecht, “THz Diode Technology:
Status, Prospects, and Applications,” _Proceedings of the IEEE_, vol. 105,
no. 6, pp. 990–1007, 2017.

[37] H.-J. Song, K. Ajito, Y. Muramoto, A. Wakatsuki, T. Nagatsuma, and
N. Kukutsu, “Uni-travelling-carrier Photodiode Module Generating 300
GHz Power Greater than 1 mW,” _IEEE Microwave and Wireless Compo-_
_nents Letters_, vol. 22, no. 7, pp. 363–365, 2012.

[38] S.-W. Huang _et al._, “Globally Stable Microresonator Turing Pattern
Formation for Coherent High-power THz Radiation On-chip,” _Physical_
_Review X_, vol. 7, no. 4, p. 041002, 2017.

[39] T. Nagatsuma, G. Ducournau, and C. C. Renaud, “Advances in Terahertz
Communications Accelerated by Photonics,” _Nature Photonics_, vol. 10,
no. 6, p. 371, 2016.

[40] Q. Lu, D. Wu, S. Sengupta, S. Slivken, and M. Razeghi, “Room Temperature Continuous Wave, Monolithic Tunable THz Sources Based
on Highly Efficient Mid-infrared Quantum Cascade Lasers,” _Scientific_
_reports_, vol. 6, 2016.

[41] A. C. Ferrari _et al._, “Science and Technology Roadmap for Graphene,
Related Two-dimensional Crystals, and Hybrid Systems,” _Nanoscale_,
vol. 7, no. 11, pp. 4598–4810, 2015.

[42] J. M. Jornet and I. F. Akyildiz, “Graphene-based plasmonic nanotransceiver for terahertz band communication,” in _Proc. of the 8th Euro-_
_pean Conference on Antennas and Propagation (EuCAP)_ . IEEE, 2014,



pp. 492–496, U.S. Patent No. 9,397,758, July 19, 2016 (Priority Date:
Dec. 6, 2013).

[43] M. Nafari, G. R. Aizin, and J. M. Jornet, “Plasmonic hemt terahertz
transmitter based on the dyakonov-shur instability: Performance analysis
and impact of nonideal boundaries,” _Physical Review Applied_, vol. 10,
no. 6, p. 064025, 2018.

[44] B. Sensale-Rodriguez _et al._, “Broadband graphene terahertz modulators
enabled by intraband transitions,” _Nature communications_, vol. 3, p. 780,
2012.

[45] P. K. Singh, G. Aizin, N. Thawdar, M. Medley, and J. M. Jornet,
“Graphene-based plasmonic phase modulator for terahertz-band communication,” in _Proc. of the 10th European Conference on Antennas and_
_Propagation (EuCAP)_ . IEEE, 2016, pp. 1–5.

[46] I. Llatser, C. Kremers, A. Cabellos-Aparicio, J. M. Jornet, E. Alarcón,
and D. N. Chigrin, “Graphene-based nano-patch antenna for terahertz radiation,” _Photonics and Nanostructures-Fundamentals and Applications_,
vol. 10, no. 4, pp. 353–358, 2012.

[47] J. M. Jornet and I. F. Akyildiz, “Graphene-based plasmonic nano-antenna
for terahertz band communication in nanonetworks,” _IEEE J. Sel. Areas_
_Commun._, vol. 31, no. 12, pp. 685–694, 2013, U.S. Patent No. 9,643,841,
May 9, 2017 (Priority Date: Dec. 6, 2013).

[48] S. V. Hum and J. Perruisseau-Carrier, “Reconfigurable reflectarrays and
array lenses for dynamic antenna beam control: A review,” _IEEE Trans._
_Antennas Propag._, vol. 62, no. 1, pp. 183–198, 2013.

[49] A. Singh, M. Andrello, N. Thawdar, and J. M. Jornet, “Design and
operation of a Graphene-based Plasmonic nano-antenna array for communication in the Terahertz band,” _IEEE J. Sel. Areas Commun. (to_
_appear)_, 2020.

[50] C. Jastrow _et al._, “Wireless Digital Data Transmission at 300 GHz,”
_Electron. Lett._, vol. 46, no. 9, pp. 661–663, 2010.

[51] L. Moeller, J. Federici, and K. Su, “2.5 Gbit/s Duobinary Signalling
with Narrow Bandwidth 0.625 Terahertz Source,” _Electron. lett._, vol. 47,
no. 15, pp. 856–858, 2011.

[52] I. Kallfass, J. Antes, D. Lopez-Diaz, S. Wagner, A. Tessmann, and
A. Leuther, “Broadband Active Integrated Circuits for Terahertz Communication,” in _European Wireless 2012; 18th European Wireless Con-_
_ference 2012_ . VDE, 2012, pp. 1–5.

[53] W. R. Deal _et al._, “A 666 GHz Demonstration Crosslink with 9.5 Gbps
Data Rate,” in _2017 IEEE MTT-S International Microwave Symposium_
_(IMS)_ . IEEE, 2017, pp. 233–235.

[54] T. Merkle _et al._, “Testbed for Phased Array Communications from 275
to 325 GHz,” in _2017 IEEE Compound Semiconductor Integrated Circuit_
_Symposium (CSICS)_ . IEEE, 2017, pp. 1–4.

[55] P. Sen, D. A. Pados, S. N. Batalama, E. Einarsson, J. P. Bird, and J. M.
Jornet, “The teranova platform: An integrated testbed for ultra-broadband
wireless communications at true terahertz frequencies,” _Comput. Netw._,
vol. 179, p. 107370, 2020.

[56] C. Belem-Goncalves _et al._, “300 ghz quadrature phase shift keying and
qam16 56 gbps wireless data links using silicon photonics photodiodes,”
_Electron. Lett._, vol. 55, no. 14, pp. 808–810, 2019.

[57] S. Koenig _et al._, “Wireless Sub-THz Communication System with High
Data Rate,” _Nature photonics_, vol. 7, no. 12, p. 977, 2013.

[58] J. M. Jornet and I. F. Akyildiz, “Channel modeling and capacity analysis
of electromagnetic wireless nanonetworks in the Terahertz band,” _IEEE_
_Trans. Wireless Commun._, vol. 10, no. 10, pp. 3211–3221, Oct. 2011.

[59] Y. Xing and T. S. Rappaport, “Propagation measurement system and
approach at 140 Ghz-moving to 6G and above 100 Ghz,” in _IEEE Global_
_Commun. Conf. (GLOBECOM)_ . IEEE, 2018, pp. 1–6.

[60] N. A. Abbasi, A. F. Molisch, and J. C. Zhang, “Measurement of directionally resolved radar cross section of human body for 140 and 220 Ghz
bands,” in _IEEE Wireless Commun. Netw. Conf. Workshops (WCNCW)_ .
IEEE, 2020, pp. 1–4.

[61] I. Kallfass _et al._, “64 gbit/s transmission over 850 m fixed wireless link at
240 ghz carrier frequency,” _Journal of Infrared, millimeter, and terahertz_
_waves_, vol. 36, no. 2, pp. 221–233, 2015.

[62] S. Priebe, C. Jastrow, M. Jacob, T. Kleine-Ostmann, T. Schrader, and
T. Kurner, “Channel and propagation measurements at 300 GHz,” _IEEE_
_Trans. Antennas Propag._, vol. 59, no. 5, pp. 1688–1698, May 2011.

[63] J. Ma, N. J. Karl, S. Bretin, G. Ducournau, and D. M. Mittleman,
“Frequency-division multiplexer and demultiplexer for terahertz wireless
links,” _Nature Communications_, vol. 8, no. 1, p. 729, 2017.

[64] S. Ishii, M. Kinugawa, S. Wakiyama, S. Sayama, T. Kamei _et al._, “Rain
attenuation in the microwave-to-terahertz waveband,” _Wireless Engineer-_
_ing and Technology_, vol. 7, no. 02, p. 59, 2016.



VOLUME 4, 2021 41


[65] ITU-R, “P.838-3: Specific attenuation model for rain for use in prediction
methods,” ITU Recommendations, Tech. Rep., 2005.

[66] F. Norouziari, E. Marchetti, E. Hoare, M. Gashinova, C. Constantinou,
P. Gardner, and M. Cherniakov, “Low-THz wave snow attenuation,” in
_Int. Conf. Radar (RADAR)_, 2018.

[67] J. Ma, J. Adelberg, R. Shrestha, L. Moeller, and D. M. Mittleman, “The
effect of snow on a terahertz wireless data link,” _Journal of Infrared,_
_Millimeter, and Terahertz Waves_, vol. 39, no. 6, pp. 505–508, 2018.

[68] R. Piesiewicz, C. Jansen, D. Mittleman, T. Kleine-Ostmann, M. Koch,
and T. Kurner, “Scattering analysis for the modeling of THz communication systems,” _IEEE Trans. Antennas Propag._, vol. 55, no. 11, pp.
3002–3009, Nov. 2007.

[69] C. Jansen, R. Piesiewicz, D. Mittleman, T. Kurner, and M. Koch, “The
impact of reflections from stratified building materials on the wave
propagation in future indoor terahertz communication systems,” _IEEE_
_Trans. Antennas Propag._, vol. 56, no. 5, pp. 1413–1419, May 2008.

[70] C. Jansen, S. Priebe, C. Moller, M. Jacob, H. Dierke, M. Koch, and
T. Kurner, “Diffuse scattering from rough surfaces in THz communication channels,” _IEEE Trans. THz Sci. Technol._, vol. 1, no. 2, pp. 462–472,
2011.

[71] J. Kokkoniemi, V. Petrov, D. Moltchanov, J. Lehtomaki, Y. Koucheryavy,
and M. Juntti, “Wideband terahertz band reflection and diffuse scattering
measurements for beyond 5G indoor wireless networks,” in _Prof. of_
_European Wireless Conference_, May 2016.

[72] S. Priebe and T. Kurner, “Stochastic modeling of THz indoor radio
channels,” _IEEE Tran. Wireless Commun._, vol. 12, no. 9, pp. 4445–4455,
2013.

[73] C. Han, A. O. Bicen, and I. Akyildiz, “Multi-ray channel modeling and
wideband characterization for wireless communications in the Terahertz

band,” _IEEE Trans. Wireless Commun._, vol. 14, no. 5, pp. 2402–2412,
May 2015.

[74] S. Kim and A. Zajic, “Statistical modeling and simulation of shortrange device-to-device communication channels at sub-THz frequencies,” _IEEE Tran. Wireless Commun._, vol. 15, no. 9, pp. 6423–6433,
September 2016.

[75] Z. Hossain, C. Mollica, and J. M. Jornet, “Stochastic multipath channel
modeling and power delay profile analysis for terahertz-band communication,” in _Proceedings of the 4th ACM International Conference on_
_Nanoscale Computing and Communication_, 2017, pp. 1–7.

[76] H. Elayan, O. Amin, B. Shihada, R. M. Shubair, and M. Alouini,
“Terahertz band: The last piece of rf spectrum puzzle for communication
systems,” _IEEE Open J. Commun. Soc._, vol. 1, pp. 1–32, Nov. 2020.

[77] “Mobile and wireless communications enablers for twenty-twenty information society (METIS),” METIS project, Tech. Rep. ICT-317669METIS/D2.1, 2014.

[78] H. Sawada, H. Nakase, S. Kato, M. Umehira, K. Sato, and H. Harada,
“Impulse response model and parameters for indoor channel modeling at
60 GHz,” in _Proc. IEEE Veh. Technol. Conf. (VTC-Spring)_, May 2010.

[79] S. Hur _et al._, “Proposal on millimeter-wave channel modeling for 5G
cellular system,” _IEEE J. Sel. Topics Signal Process._, vol. 10, no. 3, pp.
454–469, Apr. 2016.

[80] T. S. Rappaport, Y. Xing, G. R. MacCartney, A. F. Molisch, E. Mellios,
and J. Zhang, “Overview of millimeter wave communications for fifthgeneration (5G) wireless networks-with a focus on propagation models,”
_IEEE Trans. Antennas Propag._, vol. 65, no. 12, pp. 6213–6230, Dec.
2017.

[81] W. Fan, I. Carton, P. Kyösti, and G. F. Pedersen, “Emulating ray-tracing
channels in multiprobe anechoic chamber setups for virtual drive testing,”
_IEEE Trans. Antennas Propag._, vol. 64, no. 2, pp. 730–739, Feb. 2016.

[82] C. Wang, J. Bian, J. Sun, W. Zhang, and M. Zhang, “A survey of
5G channel measurements and models,” _IEEE Commun. Surveys Tuts._,
vol. 20, no. 4, pp. 3142–3168, 2018.

[83] K. Guan _et al._, “Channel characterization for intra-wagon communication
at 60 and 300 GHz bands,” _IEEE Trans. Veh. Technol._, vol. 68, no. 6, pp.
5193–5207, June 2019.

[84] S. Rey, J. M. Eckhardt, B. Peng, K. Guan, and T. Kürner, “Channel
sounding techniques for applications in THz communications: A first
correlation based channel sounder for ultra-wideband dynamic channel
measurements at 300 GHz,” in _International Congress on Ultra Modern_
_Telecommunications and Control Systems and Workshops (ICUMT)_, Nov.
2017.

[85] K. Guan, Z. Zhong, B. Ai, and T. Kürner, “Deterministic propagation
modeling for the realistic high-speed railway environment,” in _Proc._
_IEEE Veh. Technol. Conf. (VTC-Spring)_, Jun. 2013.



N. Rajatheva _et al._ : Scoring the Terabit/s Goal: Broadband Connectivity in 6G


[86] J. Nuckelt, T. Abbas, F. Tufvesson, C. Mecklenbrauker, L. Bernado, and
T. Kürner, “Comparison of ray tracing and channel-sounder measurements for vehicular communications,” in _Proc. IEEE Veh. Technol. Conf._
_(VTC-Spring)_, June 2013.

[87] S. Priebe and T. Kürner, “Stochastic modeling of THz indoor radio
channels,” _IEEE Trans. Wireless Commun._, vol. 12, no. 9, pp. 4445–4455,
Sept. 2013.

[88] K. Guan _et al._, “Towards realistic high-speed train channels at 5G
millimeter-wave band – part I: Paradigm, significance analysis, and
scenario reconstruction,” _IEEE Trans. Veh. Technol._, vol. 67, no. 10, pp.
9129–9144, Aug. 2018.

[89] K. Guan _et al._, “Towards realistic high-speed train channels at 5G
millimeter-wave band – part II: Case study for paradigm implementation,” _IEEE Transactions on Vehicular Technology_, vol. 67, no. 10, pp.
9129–9144, Aug. 2018.

[90] D. He, B. Ai, K. Guan, L. Wang, Z. Zhong, and T. Kürner, “The design
and applications of high-performance ray-tracing simulation platform for
5G and beyond wireless communications: A tutorial,” _IEEE Commun._
_Surveys Tuts._, vol. 21, no. 1, pp. 10–27, 2019.

[91] J. Ma _et al._, “Security and eavesdropping in terahertz wireless links,”
_Nature_, vol. 563, no. 7729, pp. 89–93, 2018.

[92] I. F. Akyildiz, C. Han, and S. Nie, “Combating the distance problem in the
millimeter wave and Terahertz frequency bands,” _IEEE Commun. Mag._,
vol. 56, no. 6, pp. 102–108, June 2018.

[93] I. F. Akyildiz and J. M. Jornet, “Realizing ultra-massive MIMO (1024 _×_
1024) communication in the (0.06–10) Terahertz band,” _Nano Commun._
_Netw._, vol. 8, pp. 46–54, June 2016.

[94] L. Yan, C. Han, and J. Yuan, “A dynamic array of sub-array architecture
for hybrid precoding in the millimeter wave and Terahertz bands,” in
_Proc. IEEE Int. Conf. Commun. (ICC)_, May 2019.

[95] Hadi Sarieddeen, Mohamed-Slim Alouini, Tareq Y. Al-Naffour,
“Terahertz-band ultra-massive spatial modulation MIMO,” _IEEE J. Sel._
_Areas Commun._, vol. 37, no. 9, pp. 2040–2052, Sept. 2019.

[96] S. Nie, J. M. Jornet, and I. F. Akyildiz, “Intelligent environments based on
ultra-massive MIMO platforms for wireless communication in millimeter
wave and Terahertz bands,” in _Proc. IEEE Int. Conf. Acoust., Speech, and_
_Signal Process. (ICASSP)_, Apr. 2019.

[97] X. Ma, Z. Chen, W. Chen, Y. Chi, Z. Li, C. Han, and Q. Wen, “Intelligent
reflecting surface enhanced indoor Terahertz communication systems,”
_Nano Commun. Netw._, vol. 24, pp. 1–9, May 2020.

[98] A. Singh, M. Andrello, E. Einarsson, N. Thawdar, and J. M. Jornet,
“Design and operation of a smart Graphene-Metal hybrid reflectarray at
THz frequencies,” in _Proc. Eur. Conf. Antennas Propag. (EuCAP)_, Mar.
2020.

[99] C. Han and I. F. Akyildiz, “Distance-aware bandwidth-adaptive resource
allocation for wireless systems in the Terahertz band,” _IEEE Trans. THz_
_Sci. Technol._, vol. 6, no. 4, pp. 541–553, July 2016.

[100] Z. Hossain and J. M. Jornet, “Hierarchical bandwidth modulation for
ultra-broadband Terahertz communications,” in _Proc. IEEE Int. Conf._
_Commun. (ICC)_, May 2019.

[101] Q. Xia, Z. Hossain, M. J. Medley, and J. M. Jornet, “A link-layer
synchronization and medium access control protocol for Terahertz-band
communication networks,” _IEEE Trans. Mobile Comput. (to appear)_,
2020.

[102] Q. Xia and J. M. Jornet, “Expedited neighbor discovery in directional
Terahertz communication networks enhanced by antenna side-lobe information,” _IEEE Trans. Veh. Technol._, vol. 68, no. 8, pp. 7804–7814, Aug.
2019.

[103] M. Agiwal et al., “Next Generation 5G Wireless Networks: A Comprehensive Survey," _IEEE Communications Surveys & Tutorials_, vol. 18, no.
3, Aug. 2016.

[104] M. Uysal et al., _Eds. Optical Wireless Communications- An Emerging_
_Technology_, Springer 2016.

[105] M.-A. Khalighi and M. Uysal, “Survey on Free Space Optical Communication: A Communication Theory Perspective”, _IEEE Communications_
_Surveys & Tutorials,_ vol.16, no.8, pp. 2231-2258, Nov 2014.

[106] W. Fawaz, C. Abou-Rjeily, C. Assi, “UAV-Aided Cooperation for FSO
Communication Systems,” _IEEE Commun. Mag.,_ vol. 56, no.1, Jan. 2018.

[107] P. H. Pathak, X. Feng, P. Hu, and P. Mohapatra, “Visible light communication, networking, and sensing: A survey, potential and challenges,"
_IEEE Commun. Surv. Tut.,_ vol. 17, no. 4, pp. 20472077, 2015.

[108] H. Haas, L. Yin, and C. Chen, “What is LiFi?, _J. Lightw. Technol.,_ vol.
34, no. 6, pp. 15331544, Dec. 2015.



42 VOLUME 4, 2021


N. Rajatheva _et al._ : Scoring the Terabit/s Goal: Broadband Connectivity in 6G


[109] M. S. Saud and M. Katz, “Implementation of a Hybrid Optical-RF
Wireless Network with Fast Network Handover,” Eur. Wireless, 2017.

[110] I. Ahmed, H. Karvonen, T. Kumpuniemi, et al. “Wireless Communications for the Hospital of the Future: Requirements, Challenges and
Solutions”, Int J. Wireless Inf. Networks 27, 4-–17 2020.

[111] M. Katz and D. O’Brien, “Exploiting novel concepts for visible light
communications: from light-based IoT to living surfaces”, _Optik – In-_
_ternational Journal for Light and Electron Optics_ 195:163176, DOI:
10.1016/j.ijleo.2019.163176, Aug. 2019.

[112] M. Katz and I. Ahmed, “Opportunities and Challenges for Visible Light
Communications in 6G," _Proc. 6G Wireless Summit (6G SUMMIT)_, Levi,
Finland, 2020, doi: 10.1109/6GSUMMIT49458.2020.9083805.

[113] F. Miramirkhani, M. Uysal, and E. Panayirci, “Novel channel models for
visible light communications", **Invited Paper**, SPIE Photonics West, San
Francisco, CA, USA, Feb. 2015.

[114] F. Miramirkhani, O. Narmanlioglu, M. Uysal and E. Panayirci, “A mobile
channel model for VLC and application to adaptive system design", _IEEE_
_Commun. Lett._ vol. 21, no. 5, pp. 1089-1038, Jan. 2017.

[115] M. Uysal, F. Miramirkhani, O. Narmanlioglu, T. Baykas, and E.
Panayirci, “IEEE 802.15.7r1 reference channel models for visible light
communications", _IEEE Commun. Mag.,_ vol. 55, no. 19, pp. 212217, Jan.
2017.

[116] T. Cogalan, H. Haas and E. Panayirci, “Optical spatial modulation design,” _The Royal Soc. Philos. Trans. A,_ A 378: 20190195.
http://dx.doi.org/10.1098/rsta.2019.0195, pp.1-18, Feb 2020.

[117] A. Yesilkaya, E. Basar, F. Miramirkhani, E. Panayirci, M. Uysal, H. Haas,
“Optical MIMO-OFDM with generalized LED index modulation”, _IEEE_
_Trans. Commun.,_ vol.65, no. 8, pp. 3429-3441, May 2017.

[118] E. Panayirci, A. Yesilkaya, T. Cogalan, H. Haas, H, V. Poor, “Physicallayer security with generalized space shift keying”, _IEEE Trans. Com-_
_mun,_ vol. 68, no.5, pp. 3042 – 3056, May 2020.

[119] A. Arafa, E. Panayirci, H. V. Poor, “Relay-aided secure broadcasting for
visible light communications”, _IEEE Trans. Commun,_ vol. 67, no. 6, pp.
4227-4239, Feb. 2019.

[120] A. Yesilkaya, T. Cogalan, E. Panayirci, H. Haas and H. V. Poor, **(Best**
**paper award)** “ Achieving Minimum Error in PAM Based MISO Optical
Spatial Modulation Systems", Proc. IEEE Int. Conf. Commun. (ICC),
2018.

[121] F. E. Basar, E. Panayirci, M. Uysal, H. Haas, “Generalized LED Index
Modulation Optical OFDM for MIMO Visible Light Communications
Systems, **(Best paper award)** Proc. IEEE Int. Conf. Commun. (ICC),
2016.

[122] S. Rajagopal, R. D.Roberts, S. K. Lim, “IEEE 802.15. 7 visible light communication: modulation schemes and dimming support. _IEEE Commun._
_Mag._, vol.50, no. 3, pp.72—82, 2012.

[123] Ahmed I., Bykov A., Popov A., Meglinski I., and Katz M., “Optical
Wireless Data Transfer Through Biotissues: Practical Evidence and Initial Results”, _2019 BodyNets_, Florence, Italy, Oct. 2019.

[124] I. Ahmed, A. Bykov, A. Popov, I. Meglinski and M. Katz, “Wireless data
transfer through biological tissues using near-infrared light: Testing skull
and skin phantoms,” _Photonics West 2020_, California (USA).

[125] M. Z. Chowdhury, Md. Shahjalal, Moh. K. Hasan and Y. M. Jang,
“Review the role of optical wireless communication technologies in
5G/6G and IoT solutions: Prospects, Directions, and Challenges”, _Ap-_
_plied Science,_ no. 9, pp. 1-20, 4367; doi:10.3390/app9204367, 2019.

[126] S. Chen, R. Ma, H-H. Chen, H. Zhang, W. Meng and J. Liu, “ Machineto-machine communications in ultra-dense networks – A Survey”, _IEEE_
_Commun. Surveys Tuts.,_ vol. 19, no. 3, pp. 1478–1503, 2017.

[127] G. Cossu et al., “Gigabit-class optical wireless communication system at
indoor distances (1.5 – 4 m)," _Optics Express,_ vol. 23, no. 12, June 2015.

[128] A. Gomez, K. Shi, C. Quintana, M. Sato, G. Faulkner, B. C. Thomsen,
and D. O’Brien, “Beyond 100-gb/s indoor wide-field-of-view optical
wireless communications," _IEEE Photon. Technol. Lett.,_ vol. 27, no. 4,
pp. 367-370, 2014.

[129] A.M. Cailean, M. Dimian, “Current Challenges for Visible Light Communications Usage in Vehicle Applications: A Survey,” _IEEE Communi-_
_cations Surveys & Tutorials,_ vol. 19, no.4, 2017.

[130] H. Kaushal, G. Kaddoum, “Optical Communication in Space: Challenges
and Mitigation Techniques", _IEEE Commun. Surveys Tut.,_ vol. 19, no. 1,
Feb. 2017.

[131] M. Sharma, D. Chadha, V. Chandra, "High-altitude platform for freespace optical communication: Performance evaluation and reliability
analysis," _IEEE/OSA Journal of Optical Communications and Network-_
_ing,_ vol. 8, no. 8, Aug. 2016.




[132] R. J. Hughes, J. E. Nordholt, "Free-space communications: Quantum
space race heats up," _Nature Photonics,_ vol. 11, no. 8, Aug. 2017.

[133] T. L. Marzetta, “Noncooperative cellular wireless with unlimited numbers of base station antennas,” _IEEE Trans. Wireless Commun._, vol. 9,
no. 11, pp. 3590–3600, Nov. 2010.

[134] T. L. Marzetta, E. G. Larsson, H. Yang, and H. Q. Ngo, _Fundamentals of_
_Massive MIMO_ . Cambridge University Press, 2016.

[135] E. Björnson, J. Hoydis, and L. Sanguinetti, “Massive MIMO networks:
Spectral, energy, and hardware efficiency,” _Foundations and Trends® in_
_Signal Processing_, vol. 11, no. 3–4, pp. 154–655, 2017.

[136] T. Laas, J. A. Nossek, S. Bazzi, and W. Xu, “On the impact of the mutual
reactance on the radiated power and on the achievable rates,” _IEEE Trans._
_Circuits Syst. II_, vol. 65, no. 9, pp. 1179–1183, Sep. 2018.

[137] T. Laas, J. A. Nossek, S. Bazzi, and W. Xu, “On reciprocity in physically
consistent TDD systems with coupled antennas,” _IEEE Trans. Wireless_
_Commun._, 2020.

[138] J. W. Wallace and M. A. Jensen, “Mutual coupling in MIMO wireless
systems: A rigorous network theory analysis,” _IEEE Trans. Wireless_
_Commun._, vol. 3, no. 4, pp. 1317–1325, Jul. 2004.

[139] C. Waldschmidt, S. Schulteis, and W. Wiesbeck, “Complete RF system
model for analysis of compact MIMO arrays,” _IEEE Trans. Veh. Technol._,
vol. 53, no. 3, pp. 579–586, May 2004.

[140] M. T. Ivrlaˇc and J. A. Nossek, “Toward a circuit theory of communication,” _IEEE Trans. Circuits Syst. I_, vol. 57, no. 7, pp. 1663–1683, Jul.
2010.

[141] S. Bazzi and W. Xu, “On the amount of downlink training in correlated
massive mimo channels,” _IEEE Trans. Signal Process._, vol. 66, no. 9, pp.
2286–2299, 2018.

[142] S. Bazzi _et al._, “Exploiting the massive mimo channel structural properties for minimization of channel estimation error and training overhead,”
_IEEE Access_, vol. 7, pp. 32 434–32 452, 2019.

[143] S. Buzzi and C. D’Andrea, “Doubly massive mmwave MIMO systems:
Using very large antenna arrays at both transmitter and receiver,” in _Proc._
_IEEE Global Commun. Conf. (GLOBECOM)_, 2016.

[144] S. Buzzi and C. D’Andrea, “Energy efficiency and asymptotic performance evaluation of beamforming structures in doubly massive MIMO
mmwave systems,” _IEEE Trans. Green Commun. and Netw._, vol. 2, no. 2,
pp. 385–396, June 2018.

[145] D.-. Phan-Huy _et al._, “Massive multiple input massive multiple output
for 5G wireless backhauling,” in _Proc. IEEE Global Commun. Conf._
_(GLOBECOM)_, 2017.

[146] S. Buzzi and C. D’Andrea, “Energy-efficient design for doubly massive
MIMO millimeter wave wireless systems,” in _Green Communications for_
_Energy-Efficient Wireless Systems and Networks_, H. Suraweera, J. Yang,
A. Zappone, and J. S. Thompson, Eds. The Institution of Engineering
and Technology (IET), 2020, ch. 2, to appear.

[147] O. Edfors and A. J. Johansson, “Is orbital angular momentum (OAM)
based radio communication an unexploited area?” _IEEE Trans. Antennas_
_Propag._, vol. 60, no. 2, pp. 1126–1131, Feb. 2012.

[148] E. Björnson and L. Sanguinetti, “Utility-based precoding optimization
framework for large intelligent surfaces,” in _Proc. Asilomar Conf. Sig-_
_nals, Syst., and Comput. (ASILOMAR)_, Nov. 2019.

[149] B. Zong, C. Fan, X. Wang, X. Duan, B. Wang, and J. Wang, “6G
technologies: Key drivers, core requirements, system architectures, and
enabling technologies,” _IEEE Veh. Technol. Mag._, vol. 14, no. 3, pp. 18–
27, Sept. 2019.

[150] M. R. Konkol, D. D. Ross, S. Shi, C. E. Harrity, A. A. Wright,
C. A. Schuetz, and D. W. Prather, “High-power photodiode-integratedconnected array antenna,” _J. Lightw. Technol._, vol. 35, no. 10, pp. 2010–
2016, May 2017.

[151] B. Xu, W. Qi, Y. Zhao, L. Wei, and C. Zhang, “Holographic radio interferometry for target tracking in dense multipath indoor environments,”
in _Proc. Int. Conf. Wireless Commun. and Signal Process. (WCSP)_, Oct.
2017.

[152] Z. Baiqing, Z. Xiaohong, L. Xiaotong, W. Jianli, and C. Yijun, “Photonics
defined radio: Concept, architecture and applications,” in _Proc. Asia_
_Commun. Photon. Conf. (ACP)_, Nov. 2017.

[153] C. Liaskos, S. Nie, A. Tsioliaridou, A. Pitsillides, S. Ioannidis, and
I. Akyildiz, “A new wireless communication paradigm through softwarecontrolled metasurfaces,” _IEEE Commun. Mag._, vol. 56, no. 9, pp. 162–
169, Sept. 2018.

[154] Q. Wu and R. Zhang, “Towards smart and reconfigurable environment:
Intelligent reflecting surface aided wireless network,” _IEEE Commun._
_Mag._, vol. 58, no. 1, pp. 106–112, Jan. 2020.



VOLUME 4, 2021 43


[155] E. Björnson, Ö. Özdogan, and E. G. Larsson, “Reconfigurable intelligent
surfaces: Three myths and two critical questions,” 2020. [Online].
[Available: https://arxiv.org/pdf/2006.03377.pdf](https://arxiv.org/pdf/2006.03377.pdf)

[156] S. Zhang, Q. Wu, S. Xu, and G. Y. Li, “Fundamental green tradeoffs:
Progresses, challenges, and impacts on 5G networks,” vol. 19, no. 1, pp.
33–56, First Quarter 2017.

[157] Q. Wu, G. Y. Li, W. Chen, D. W. K. Ng, and R. Schober, “An overview of
sustainable green 5G networks,” _IEEE Wireless Commun._, vol. 24, no. 4,
pp. 72–80, Aug. 2017.

[158] V. S. Asadchy, M. Albooyeh, S. N. Tcvetkova, A. Díaz-Rubio, Y. Ra’di,
and S. A. Tretyakov, “Perfect control of reflection and refraction using
spatially dispersive metasurfaces,” _Phys. Rev. B_, vol. 94, no. 7, Aug. 2016.

[159] D. Headland _et al._, “Terahertz reflectarrays and nonuniform metasurfaces,” _IEEE J. Sel. Topics Quantum Electron._, vol. 23, no. 4, pp. 1–18,
July 2017.

[160] Ö. Özdogan, E. Björnson, and E. G. Larsson, “Intelligent reflecting
surfaces: Physics, propagation, and pathloss modeling,” _IEEE Wireless_
_Commun. Lett. (to appear)_, 2020.

[161] C. Huang, A. Zappone, G. C. Alexandropoulos, M. Debbah, and C. Yuen,
“Reconfigurable intelligent surfaces for energy efficiency in wireless
communication,” _IEEE Trans. Wireless Commun._, vol. 18, no. 8, pp.
4157–4170, Aug. 2019.

[162] E. Björnson and L. Sanguinetti, “Power scaling laws and near-field
behaviors of massive MIMO and intelligent reflecting surfaces,” 2019.

[[Online]. Available: https://arxiv.org/pdf/2002.04960.pdf](https://arxiv.org/pdf/2002.04960.pdf)

[163] Q. Wu and R. Zhang, “Intelligent reflecting surface enhanced wireless
network via joint active and passive beamforming,” _IEEE Trans. Wireless_
_Commun._, vol. 18, no. 11, pp. 5394–5409, Nov. 2019.

[164] B. Matthiesen, E. Björnson, E. D. Carvalho, and P. Popovski,
“Intelligent reflecting surfaces that track a mobile receiver: A
continuous time propagation model,” 2020. [Online]. Available:
[https://arxiv.org/pdf/2006.06991.pdf](https://arxiv.org/pdf/2006.06991.pdf)

[165] E. Björnson, Ö. Özdogan, and E. G. Larsson, “Intelligent reflecting
surface vs. decode-and-forward: How large surfaces are needed to beat
relaying?” _IEEE Wireless Commun. Lett._, vol. 9, no. 2, pp. 244–248,
2020.

[166] Y. Liu, J. Zhao, L. Ming, and Q. Wu, “Intelligent reflecting surface
aided MISO uplink communication network: Feasibility and SINR
[optimization,” 2020. [Online]. Available: https://arxiv.org/pdf/2007.](https://arxiv.org/pdf/2007.01482.pdf)
[01482.pdf](https://arxiv.org/pdf/2007.01482.pdf)

[167] Q. Wu, S. Zhang, B. Zheng, C. You, and R. Zhang, “Intelligent reflecting
surface aided wireless communications: A tutorial,” 2020. [Online].
[Available: https://arxiv.org/pdf/2007.02759.pdf](https://arxiv.org/pdf/2007.02759.pdf)

[168] X. Guan, Q. Wu, and R. Zhang, “Joint power control and passive
beamforming in IRS-assisted spectrum sharing,” _IEEE Communications_
_Letters_, 2020.

[169] J. Yuan, Y.-C. Liang, J. Joung, G. Feng, and E. G. Larsson, “Intelligent
reflecting surface-assisted cognitive radio system,” 2019. [Online].
[Available: https://arxiv.org/pdf/1912.10678.pdf](https://arxiv.org/pdf/1912.10678.pdf)

[170] Q. Wu and R. Zhang, “Weighted sum power maximization for intelligent reflecting surface aided SWIPT,” _IEEE Wireless Communications_
_Letters_, vol. 9, no. 5, pp. 586–590, May 2020.

[171] D. Mishra and H. Johansson, “Channel estimation and low-complexity
beamforming design for passive intelligent surface assisted MISO wireless energy transfer,” in _Proc. IEEE Int. Conf. Acoust., Speech, and Signal_
_Process. (ICASSP)_, 2019.

[172] Q. Wu and R. Zhang, “Joint active and passive beamforming optimization
for intelligent reflecting surface assisted SWIPT under QoS constraints,”
_IEEE J. Sel. Areas Commun._, to appear, 2020.

[173] X. Guan, Q. Wu, and R. Zhang, “Intelligent reflecting surface assisted
secrecy communication: Is artificial noise helpful or not?” _IEEE Wireless_
_Communications Letters_, 2020.

[174] H. Yang, Z. Xiong, J. Zhao, D. Niyato, Q. Wu, H. V. Poor, and
M. Tornatore, “Intelligent reflecting surface assisted anti-jamming
communications: A fast reinforcement learning approach,” 2020.

[[Online]. Available: https://arxiv.org/pdf/2004.12539.pdf](https://arxiv.org/pdf/2004.12539.pdf)

[175] X. Lu, W. Yang, X. Guan, Q. Wu, and Y. Cai, “Robust and secure
beamforming for intelligent reflecting surface aided mmwave MISO
[systems,” 2020. [Online]. Available: https://arxiv.org/pdf/2003.11195.pdf](https://arxiv.org/pdf/2003.11195.pdf)

[176] H. Yang, Z. Xiong, J. Zhao, D. Niyato, L. Xiao, and Q. Wu,
“Deep reinforcement learning based intelligent reflecting surface
for secure wireless communications,” 2020. [Online]. Available:
[https://arxiv.org/pdf/2002.12271.pdf](https://arxiv.org/pdf/2002.12271.pdf)



N. Rajatheva _et al._ : Scoring the Terabit/s Goal: Broadband Connectivity in 6G


[177] B. Zheng, Q. Wu, and R. Zhang, “Intelligent reflecting surface-assisted
multiple access with user pairing: NOMA or OMA?” _IEEE Commun._
_Lett._, vol. 24, no. 4, pp. 753–757, 2020.

[178] Y. Li, M. Jiang, Q. Zhang, and J. Qin, “Joint beamforming
design in multi-cluster MISO NOMA intelligent reflecting surfaceaided downlink communication networks,” 2019. [Online]. Available:
[https://arxiv.org/pdf/1909.06972.pdf](https://arxiv.org/pdf/1909.06972.pdf)

[179] A. U. Makarfi, K. M. Rabie, O. Kaiwartya, K. Adhikari, X. Li, M. QuirozCastellanos, and R. Kharel, “Reconfigurable intelligent surfaces-enabled
vehicular networks: A physical layer security perspective.” [Online].
[Available: https://arxiv.org/pdf/2004.11288.pdf](https://arxiv.org/pdf/2004.11288.pdf)

[180] A. U. Makarfi, K. M. Rabie, O. Kaiwartya, X. Li, and R. Kharel,
“Physical layer security in vehicular networks with reconfigurable
[intelligent surfaces.” [Online]. Available: https://arxiv.org/pdf/1912.](https://arxiv.org/pdf/1912.12183.pdf)
[12183.pdf](https://arxiv.org/pdf/1912.12183.pdf)

[181] M. Hua, Q. Wu, D. W. K. Ng, J. Zhao, and L. Yang, “Intelligent reflecting
surface-aided joint processing coordinated multipoint transmission,”
[2020. [Online]. Available: https://arxiv.org/pdf/2003.13909.pdf](https://arxiv.org/pdf/2003.13909.pdf)

[182] Q. Zhang, W. Saad, and M. Bennis, “Reflections in the sky: Millimeter
wave communication with UAV-carried intelligent reflectors,” in _Proc._
_GLOBECOM_ . IEEE, 2019, pp. 1–6.

[183] M. Hua, L. Yang, Q. Wu, C. Pan, C. Li, and A. L. Swindlehurst,
“UAV-assisted intelligent reflecting surface symbiotic radio system,”
[2020. [Online]. Available: https://arxiv.org/abs/2007.14029](https://arxiv.org/abs/2007.14029)

[184] D. Ma, M. Ding, and M. Hassan, “Enhancing cellular communications for
UAVs via intelligent reflective surface,” in _Proc. IEEE WCNC_ . IEEE,
2020, pp. 1–6.

[185] S. Li, B. Duo, X. Yuan, Y.-C. Liang, and M. Di Renzo, “Reconfigurable
intelligent surface assisted uav communication: Joint trajectory design
and passive beamforming,” _IEEE Wireless Communications Letters_,
2020.

[186] Y. Zeng, Q. Wu, and R. Zhang, “Accessing from the sky: A tutorial on
UAV communications for 5G and beyond,” _Proceedings of the IEEE_, vol.
107, no. 12, pp. 2327–2375, 2019.

[187] S. Abeywickrama, R. Zhang, Q. Wu, and C. Yuen, “Intelligent reflecting
surface: Practical phase shift model and beamforming optimization.”

[[Online]. Available: https://arxiv.org/pdf/2002.10112.pdf](https://arxiv.org/pdf/2002.10112.pdf)

[188] Q. Wu and R. Zhang, “Beamforming optimization for intelligent reflecting surface with discrete phase shifts,” in _Proc. IEEE International_
_Conference on Acoustics, Speech and Signal Processing (ICASSP)_, 2019,
pp. 7830–7833.

[189] Q. Wu and R. Zhang, “Beamforming optimization for wireless network
aided by intelligent reflecting surface with discrete phase shifts,” _IEEE_
_Trans. Commun._, vol. 68, no. 3, pp. 1838–1851, Mar. 2020.

[190] M.-M. Zhao, Q. Wu, M.-J. Zhao, and R. Zhang, “Two-timescale beamforming optimization for intelligent reflecting surface enhanced wireless
network,” in _Proc. IEEE Sensor Array and Multichannel Signal Process-_
_ing Workshop (SAM)_, 2020, pp. 1–5.

[191] M.-M. Zhao, Q. Wu, M.-J. Zhao, and R. Zhang, “Exploiting
amplitude control in intelligent reflecting surface aided wireless
communication with imperfect CSI,” 2020. [Online]. Available:
[https://arxiv.org/pdf/2005.07002.pdf](https://arxiv.org/pdf/2005.07002.pdf)

[192] M.-M. Zhao, Q. Wu, M.-J. Zhao, and R. Zhang, “Intelligent reflecting surface enhanced wireless network: Twotimescale beamforming optimization,” 2019. [Online]. Available:
[https://arxiv.org/pdf/1912.01818.pdf](https://arxiv.org/pdf/1912.01818.pdf)

[193] X. Guan, Q. Wu, and R. Zhang, “Anchor-assisted intelligent reflecting
surface channel estimation for multiuser communications,” 2020.

[[Online]. Available: https://arxiv.org/abs/2008.00622](https://arxiv.org/abs/2008.00622)

[194] B. Zheng and R. Zhang, “Intelligent reflecting surface-enhanced OFDM:
Channel estimation and reflection optimization,” _IEEE Wireless Commu-_
_nication Letters_, vol. 9, no. 4, pp. 518–522, 2020.

[195] A. Taha, M. Alrabeiah, and A. Alkhateeb, “Enabling large intelligent
surfaces with compressive sensing and deep learning,” 2019. [Online].
[Available: https://arxiv.org/pdf/1904.10136.pdf](https://arxiv.org/pdf/1904.10136.pdf)

[196] H. Q. Ngo, A. Ashikhmin, H. Yang, E. G. Larsson, and T. L. Marzetta,
“Cell-free massive MIMO: Uniformly great service for everyone,” in
_Proc. IEEE Int. Workshop Signal Process. Adv. in Wireless Commun._
_(SPAWC)_ . IEEE, July 2015.

[197] H. Q. Ngo, A. Ashikhmin, H. Yang, E. G. Larsson, and T. L. Marzetta,
“Cell-free massive MIMO versus small cells,” _IEEE Trans. Wireless_
_Commun._, vol. 16, no. 3, pp. 1834–1850, Mar. 2017.



44 VOLUME 4, 2021


N. Rajatheva _et al._ : Scoring the Terabit/s Goal: Broadband Connectivity in 6G


[198] E. Nayebi, A. Ashikhmin, T. L. Marzetta, H. Yang, and B. D. Rao,
“Precoding and power optimization in cell-free Massive MIMO systems,”
_IEEE Trans. Wireless Commun._, vol. 16, no. 7, pp. 4445–4459, July 2017.

[199] S. Buzzi, and C. D’Andrea, “Cell-free massive MIMO: User-centric
approach,” _IEEE Wireless Commun. Lett._, Vol. 6, pp. 706–709, Dec.
2017.

[200] M. Boldi _et al._, “Coordinated multipoint (CoMP) systems,” in _Mobile and_
_Wireless Communications for IMT-Advanced and Beyond_ . Wiley, 2011.

[201] E. Björnson and E. Jorswieck, “Optimal resource allocation in coordinated multi-cell systems,” _Foundations and Trends® in Commun. Inf._
_Theory_, vol. 9, no. 2–3, pp. 113–381, 2013.

[202] G. Interdonato, E. Björnson, H. Q. Ngo, P. Frenger, and E. G. Larsson,
“Ubiquitous cell-free massive MIMO communications,” _J. Wireless Com-_
_mun. Netw._, vol. 2019, no. 1, pp. 197–209, Aug. 2019.

[203] J. Liu _et al._, “Initial access, mobility, and user-centric multi-beam operation in 5G New Radio,” _IEEE Commun. Mag._, vol. 56, no. 3, pp. 35–41,
Mar. 2018.

[204] S. Buzzi, C. D’Andrea, A. Zappone, and C. D’Elia, “User-centric 5G
cellular networks: Resource allocation and comparison with the cellfree massive MIMO approach,” _IEEE Trans. Wireless Commun._, vol. 19,
no. 2, pp. 1250–1264, Feb. 2020.

[205] E. Björnson and L. Sanguinetti, “Scalable cell-free massive MIMO
systems,” _IEEE Trans. Commun._, 2020.

[206] C. D’Andrea, A. Garcia-Rodriguez, G. Geraci, L. G. Giordano, and
S. Buzzi, “Cell-free massive MIMO for UAV communications,” in _Proc._
_IEEE Int. Conf. Commun. (ICC)_, May 2019.

[207] C. D’Andrea, A. Garcia-Rodriguez, G. Geraci, L. G. Giordano, and
S. Buzzi, “Analysis of UAV communications in cell-free massive MIMO
systems,” _IEEE Open J. Commun. Soc._, Jan. 2020.

[208] E. Björnson and L. Sanguinetti, “Making cell-free massive MIMO competitive with MMSE processing and centralized implementation,” _IEEE_
_Trans. Wireless Commun._, vol. 19, no. 1, pp. 77–90, Jan. 2020.

[209] 3GPP TS 38.300, _NR; Overall description; Stage-2_, Release 15.

[210] 3GPP TR 38.812, _Study on Non-Orthogonal Multiple Access (NOMA)_
_for NR_, Release 15.

[211] Z. Yuan, Y. Hu, W. Li, and J. Dai, “Blind multi-user detection for autonomous grant-free high-overloading multiple-access without reference
signal,” in _Proc. IEEE Veh. Technol. Conf. (VTC-Spring)_, July 2018.

[212] J. Kaleva, A. Tölli, M. Juntti, R. A. Berry, and M. L. Honig, “Decentralized joint precoding with pilot-aided beamformer estimation,” _IEEE_
_Trans. Signal Process._, vol. 66, no. 9, pp. 2330–2341, May 2018.

[213] E. Nayebi, A. Ashikhmin, T. L. Marzetta, and B. D. Rao, “Performance
of cell-free massive MIMO systems with MMSE and LSFD receivers,”
in _Proc. Asilomar Conf. Signals, Syst., and Comput. (ASILOMAR)_, Nov.
2016.

[214] K. Li, C. Jeon, J. R. Cavallaro, and C. Studer, “Feedforward architectures
for decentralized precoding in massive MU-MIMO systems,” in _Proc._
_Asilomar Conf. Signals, Syst., and Comput. (ASILOMAR)_, Nov. 2018.

[215] A. Tölli _et al._, “Distributed coordinated transmission with forwardbackward training for 5G radio access,” _IEEE Commun. Mag._, vol. 57,
no. 1, pp. 58–64, Jan. 2019.

[216] I. Atzeni, B. Gouda, and A. Tölli, “Distributed precoding design via
over-the-air signaling for cell-free massive MIMO,” _IEEE Trans. Wireless_
_Commun._, vol. 2, no. 20, pp. 1201–1216, Feb. 2021.

[217] I. Atzeni, B. Gouda, and A. Tölli, “Distributed joint receiver design for
uplink cell-free massive MIMO,” in _Proc. IEEE Int. Conf. Commun._
_(ICC)_, June 2020.

[218] C. Czegledi _et al._, “Demonstrating 139 Gbps and 55.6 bps/Hz spectrum
efficiency using 8 _×_ 8 MIMO over a 1.5 km link at 73.5 GHz,” in _Proc._
_IEEE Int. Symp. Microw. (IMS)_, Aug. 2020.

[219] H. Dahrouj, A. Douik, F. Rayal, T. Y. Al-Naffouri, and M. Alouini, “Costeffective hybrid RF/FSO backhaul solution for next generation wireless
systems,” _IEEE Wireless Commun. Mag._, vol. 22, no. 5, pp. 98–104, Oct.
2015.

[220] Ericsson, “Mobile data traffic growth outlook.” [On[line]. Available: https://www.ericsson.com/en/mobility-report/reports/](https://www.ericsson.com/en/mobility-report/reports/november-2018/mobile-data-traffic-growth-outlook.)
[november-2018/mobile-data-traffic-growth-outlook.](https://www.ericsson.com/en/mobility-report/reports/november-2018/mobile-data-traffic-growth-outlook.)

[221] C. Madapatha, B. Makki, C. Fang, O. Teyeb, E. Dahlman, M.S. Alouini, and T. Svensson, “On integrated access and backhaul
networks: Current status and potentials,” 2020. [Online]. Available:
[https://arxiv.org/pdf/2006.14216.pdf](https://arxiv.org/pdf/2006.14216.pdf)

[222] O. Teyeb, A. Muhammad, G. Mildh, E. Dahlman, F. Barac, and B. Makki,
“Integrated Access Backhauled Networks,” in _Proc. IEEE Veh. Technol._
_Conf. (VTC-Fall)_, Sept. 2019.




[223] 3GPP, “Overview of 3GPP Release 10,” 3rd Generation Partnership
Project (3GPP), Tech. Rep., 06 2016, v0.2.1.

[224] S. M. Azimi-Abarghouyi, B. Makki, M. Haenggi, M. Nasiri-Kenari, and
T. Svensson, “Coverage analysis of finite cellular networks: A stochastic
geometry approach,” in _2018 Iran Workshop on Communication and_
_Information Theory (IWCIT)_, Apr. 2018.

[225] S. M. Azimi-Abarghouyi, B. Makki, M. Nasiri-Kenari, and T. Svensson,
“Stochastic Geometry Modeling and Analysis of Finite Millimeter Wave
Wireless Networks,” _IEEE Trans. Veh. Technol._, vol. 68, no. 2, pp. 1378–
1393, Feb. 2019.

[226] T. S. Rappaport, Y. Xing, G. R. MacCartney, A. F. Molisch, E. Mellios,
and J. Zhang, “Overview of Millimeter Wave Communications for FifthGeneration (5G) Wireless Networks—With a Focus on Propagation
Models,” _IEEE Trans. Antennas Propag._, vol. 65, no. 12, pp. 6213–6230,
Dec. 2017.

[227] B. Błaszczyszyn, “Lecture Notes on Random Geometric ModelsRandom Graphs, Point Processes and Stochastic Geometry,” 2017.

[[Online]. Available: https://hal.inria.fr/cel-01654766/document](https://hal.inria.fr/cel-01654766/document)

[228] E. J. Oughton and Z. Frias, “Exploring the cost, coverage and rollout
implications of 5G in Britain,” _Cambridge: Centre for Risk Studies,_
_Cambridge Judge Business School_, 2016.

[229] H. A. Willebrand and B. S. Ghuman, “Fiber optics without fiber,” _IEEE_
_Spectrum_, vol. 38, no. 8, pp. 40–45, Aug. 2001.

[230] “AT & T and Verizon to use Integrated Access and
Backhaul for 2021 5G networks,” Dec. 2019, available:
https://techblog.comsoc.org/2019/12/16/att-and-verizon-to-useintegrated-access-and-backhaul-for-2021-5g-networks.

[231] “Verizon to Use ’Integrated Access Backhaul’ for Fiber-Less 5G,”
Oct. 2019, Available: https://www.lightreading.com/mobile/5g/verizonto-use-integrated-access-backhaul-for-fiber-less-5g/d/d-id/754752 .

[232] X. Huang, J. A. Zhang, R. P. Liu, Y. J. Guo, and L. Hanzo, “Airplaneaided integrated networking for 6G wireless: Will it work?” _IEEE Veh._
_Technol. Mag._, vol. 14, No. 3, pp. 84–91, Sept. 2019.

[233] M. Saily, C. Barjau, D. Navratil, A. Prasad, D. Gomez-Barquero, and
F. B. Tesema, “5G Radio Access Networks: Enabling Efficient Point-toMultipoint Transmissions,” _IEEE Veh. Technol. Mag._, vol. 14, no. 4, pp.
29–37, December 2019.

[234] T. Tran _et al._, “Enabling Multicast and Broadcast in the 5G Core for
Converged Fixed and Mobile Networks,” _IEEE Trans. on Broadcast._,
vol. 66, no. 2, pp. 428–439, May 2020.

[235] J. J. Gimenez _et al._, “5G New Radio for Terrestrial Broadcast: A ForwardLooking Approach for NR-MBMS,” _IEEE Trans. on Broadcast._, vol. 65,
no. 2, pp. 356–368, June 2019.

[236] “Decision (EU) 2017/899 of the European Parliament and of the Council
of 17 May 2017 on the use of the 470-790 MHz frequency band in the
Union,” _Official Journal of the European Union_, May 2017.

[237] D. Gomez-Barquero and M. W. Caldwell, “Broadcast television spectrum
incentive auctions in the U.S.: Trends, challenges, and opportunities,”
_IEEE Commun. Mag._, vol. 53, no. 7, pp 50–56, July 2015.

[238] E. Stare, J. J. Gimenez, and P. Klenner, “WIB: A new system concept for
digital terrestrial television (DTT),” in _Proc. IBC Conf._, Sept. 2016.

[239] J. J. Gimenez, D. Gomez-Barquero, J. Morgade, and E. Stare, “Wideband
Broadcasting: A Power-Efficient Approach to 5G Broadcasting,” _IEEE_
_Commun. Mag._, vol. 56, no. 3, pp. 119–125, March 2018.

[240] “EPIC: Enabling practical wireless Tb/s communications with next
[generation channel coding.” [Online]. Available: https://epic-h2020.eu/](https://epic-h2020.eu/)

[241] C. Kestel, M. Herrmann, and N. When, “When channel coding hits
the implementation wall,” in _Proc. IEEE Int. Symp. Turbo Codes and_
_Iterative Inf. Process. (ISTC)_, Dec. 2018.

[242] A. Elkelesh, M. Ebada, S. Cammerer, and S. ten Brink, “Belief propagation decoding of polar codes on permuted factor graphs,” in _Proc. IEEE_
_Wireless Commun. and Netw. Conf. (WCNC)_, April 2018.

[243] A. Elkelesh, M. Ebada, S. Cammerer, and S. ten Brink, “Belief propagation list decoding of polar codes,” _IEEE Communications Letters_, vol. 22,
no. 8, pp. 1536–1539, Aug 2018.

[244] N. Doan, S. A. Hashemi, M. Mondelli, and W. J. Gross, “On the decoding
of polar codes on permuted factor graphs,” in _Proc. IEEE Global Com-_
_mun. Conf. (GLOBECOM)_, Dec 2018.

[245] V. Ranasinghe, N. Rajatheva, and M. Latva-aho, “Partially permuted
multi-trellis belief propagation for polar codes,” 2019. [Online].
[Available: https://arxiv.org/pdf/1911.08868.pdf](https://arxiv.org/pdf/1911.08868.pdf)

[246] G. Sarkis, P. Giard, A. Vardy, C. Thibeault, and W. J. Gross, “Fast polar
decoders: Algorithm and implementation,” _IEEE J. Sel. Areas Commun._,
vol. 32, no. 5, pp. 946–957, 2014.



VOLUME 4, 2021 45


[247] G. Sarkis, P. Giard, A. Vardy, C. Thibeault, and W. J. Gross, “Fast list
decoders for polar codes,” _IEEE J. Sel. Areas Commun._, vol. 34, no. 2,
pp. 318–328, 2016.

[248] H. Gamage, V. Ranasinghe, N. Rajatheva, and M. Latva-aho, “Low
latency decoder for short blocklength polar codes,” 2019. [Online].
[Available: https://arxiv.org/pdf/1911.03201.pdf](https://arxiv.org/pdf/1911.03201.pdf)

[249] A. Eslami and H. Pishro-Nik, “On finite-length performance of polar
codes: Stopping sets, error floor, and concatenated design,” _IEEE Trans._
_Commun._, vol. 61, no. 3, pp. 919–929, March 2013.

[250] A. Balatsoukas-Stimming, G. Karakonstantis, and A. Burg, “Enabling
complexity-performance trade-offs for successive cancellation decoding
of polar codes,” in _IEEE Int. Symp. Inf. Theory_, 2014, pp. 2977–2981.

[251] T. Gruber, S. Cammerer, J. Hoydis, and S. t. Brink, “On deep learningbased channel decoding,” in _Annu. Conf. Inf. Sci and Syst. (CISS)_, 2017.

[252] E. Nachmani, E. Marciano, L. Lugosch, W. J. Gross, D. Burshtein, and
Y. Be’ery, “Deep learning methods for improved decoding of linear
codes,” _IEEE J. Sel. Topics Signal Process._, vol. 12, no. 1, pp. 119–131,
2018.

[253] N. S. Loghin, J. Zöllner, B. Mouhouche, D. Ansorregui, J. Kim, and
S. Park, “Non-uniform constellations for atsc 3.0,” _IEEE Trans. Broad-_
_cast._, vol. 62, no. 1, pp. 197–203, 2016.

[254] G. Böcherer, F. Steiner, and P. Schulte, “Bandwidth efficient and ratematched low-density parity-check coded modulation,” _IEEE Trans. Com-_
_mun._, vol. 63, no. 12, pp. 4651–4665, Dec. 2015.

[255] M. Pikus and W. Xu, “Bit-level probabilistically shaped coded modulation,” _IEEE Commun. Lett._, vol. 21, no. 9, pp. 1929–1932, 2017.

[256] N. Ul Hassan, W. Xu, and A. Kakkavas, “Applying coded modulation
with probabilistic and geometric shaping for wireless backhaul channel,”
in _Proc. IEEE Int. Symp. Pers., Indoor and Mobile Radio Commun._
_(PIMRC)_, 2018.

[257] O. Iscan, R. Böhnke, and W. Xu, “Shaped polar codes for higher order
modulation,” _IEEE Commun. Lett._, vol. 22, no. 2, pp. 252–255, 2018.

[258] O. Iscan, R. Böhnke, and W. Xu, “Probabilistic shaping using 5G new
radio polar codes,” _IEEE Access_, vol. 7, pp. 22 579–22 587, Feb. 2019.

[259] O. Iscan, R. Böhnke, and W. Xu, “Sign-bit shaping using polar codes,”
_Trans. Emerg. Telecommun. Technol._, 2020.

[260] P. Schulte and G. Böcherer, “Constant composition distribution matching,” _IEEE Trans. Inf. Theory_, vol. 62, no. 1, pp. 430–434, 2016.

[261] O. Iscan, R. Böhnke, and W. Xu, “Polar coded distribution matching,”
_Electron. Lett._, vol. 55, no. 9, pp. 537–539, 2019.

[262] R. Böhnke, O. Iscan, and W. Xu, “Multi-level distribution matching,”
_IEEE Commun. Lett._, 2020.

[263] S. Buzzi, C. D’Andrea, T. Foggi, A. Ugolini, and G. Colavolpe, “Singlecarrier modulation versus OFDM for millimeter-wave wireless MIMO,”
_IEEE Trans. Commun._, vol. 66, no. 3, pp. 1335–1348, Mar. 2018.

[264] M. Mukherjee, L. Shu, V. Kumar, P. Kumar, and R. Matam, “Reduced
out-of-band radiation-based filter optimization for UFMC systems in
5G,” in _Proc. Int. Wireless Commun. Mob. Comp. Conf. (IWCMC)_, Aug.
2015.

[265] S. Buzzi, C. D’Andrea, D. Li, and S. Feng, “MIMO-UFMC transceiver
schemes for millimeter-wave wireless communications,” _IEEE Trans._
_Commun._, vol. 67, no. 5, pp. 3323–3336, May 2019.

[266] C. D’Andrea, S. Buzzi, D. Li, and S. Feng, “Adaptive data detection in
phase-noise impaired MIMO-UFMC systems at mmwave,” in _2018 IEEE_
_29th Annual International Symposium on Personal, Indoor and Mobile_
_Radio Communications (PIMRC)_, 2018, pp. 231–235.

[267] Y. Medjahdi, M. Terre, D. L. Ruyet, D. Roviras, and A. Dziri, “Performance analysis in the downlink of asynchronous OFDM/FBMC based
multi-cellular networks,” _IEEE Trans. Wireless Commun._, vol. 10, no. 8,
pp. 2630–2639, Aug. 2011.

[268] Y. Xin, “FB-OFDM: A novel multicarrier Scheme for 5G,” in _Proc. Eur._
_Conf. Netw. and Commun. (EuCNC)_, June. 2016.

[269] S. Bicais and J.-B. Dore, “Phase Noise Model Selection for Sub-THz
Communications,” in _Proc. IEEE Global Commun. Conf. (GLOBECOM)_,
2019.

[270] S. Bicais and J.-B. Doré, “Design of Digital Communications for Strong
Phase Noise Channels,” _IEEE Open J. Veh. Technol._, vol. 1, pp. 227–243,
2020.

[271] R. Krishnan, M. R. Khanzadi, T. Eriksson, and T. Svensson, “Soft
Metrics and Their Performance Analysis for Optimal Data Detection in
the Presence of Strong Oscillator Phase Noise,” _IEEE Trans. Commun._,
vol. 61, no. 6, pp. 2385–2395, 2013.



N. Rajatheva _et al._ : Scoring the Terabit/s Goal: Broadband Connectivity in 6G


[272] A. Sabharwal, P. Schniter, D. Guo, D. W. Bliss, S. Rangarajan, and
R. Wichman, “In-band full-duplex wireless: Challenges and opportunities,” _IEEE J. Sel. Areas Commun._, vol. 32, no. 9, pp. 1637–1652, Sept.
2014.

[273] C. Zhong, H. A. Suraweera, G. Zheng, I. Krikidis, and Z. Zhang,
“Wireless information and power transfer with full duplex relaying,”
_IEEE Trans. Commun._, vol. 62, no. 10, pp. 3447–3461, 2014.

[274] M. Mohammadi, H. A. Suraweera, G. Zheng, C. Zhong, and I. Krikidis,
“Full-duplex mimo relaying powered by wireless energy transfer,” in
_Proc. IEEE Int. Workshop Signal Process. Adv. in Wireless Commun._
_(SPAWC)_, 2015, pp. 296–300.

[275] K. M. Rabie, B. Adebisi, and M. Alouini, “Half-duplex and full-duplex
AF and DF relaying with energy-harvesting in log-normal fading,” _IEEE_
_Trans. Green Commun. and Netw._, vol. 1, no. 4, pp. 468–480, 2017.

[276] K. Rabie, B. Adebisi, G. Nauryzbayev, O. S. Badarneh, X. Li, and
M. Alouini, “Full-duplex energy-harvesting enabled relay networks in
generalized fading channels,” _IEEE Wireless Communications Letters_,
vol. 8, no. 2, pp. 384–387, 2019.

[277] G. Nauryzbayev, M. Abdallah, and K. M. Rabie, “Outage probability
of the eh-based full-duplex AF and DF relaying systems in _α −_ _µ_
environment,” in _Proc. IEEE Veh. Technol. Conf. (VTC-Fall)_, 2018.

[278] K. E. Kolodziej, B. T. Perry, and J. S. Herd, “In-band full-duplex technology: Techniques and systems survey,” _IEEE Trans. Microw. Theory_
_Tech._, vol. 67, no. 7, pp. 3025–3041, July 2019.

[279] K. Senel, H. V. Cheng, E. Björnson, and E. G. Larsson, “What role can
NOMA play in massive MIMO?” _IEEE J. Sel. Topics Signal Process._,
vol. 13, no. 3, pp. 597–611, June 2019.

[280] Y. Liu, Z. Qin, M. Elkashlan, Z. Ding, A. Nallanathan, and L. Hanzo,
“Non-orthogonal multiple access for 5G and beyond,” _Proc. IEEE_, vol.
105, pp. 2347–2381, Dec. 2017.

[281] B. Clerckx, H. Joudeh, C. Hao, M. Dai, and B. Rassouli, “Rate splitting
for MIMO wireless networks: A promising PHY-layer strategy for LTE
evolution,” _IEEE Commun. Mag._, vol. 54, no. 5, pp. 98–105, May 2016.

[282] L. Dai, B. Wang, Z. Ding, Z. Wang, S. Chen, and L. Hanzo, “A survey of
non-orthogonal multiple access for 5G,” _IEEE Commun. Surv. Tut._, vol.
20, no. 3, pp. 2294–2323, May 2018.

[283] S. M. R. Islam, N. Avazov, O. A. Dobre, and K.-S. Kwak, “Power-domain
non-orthogonal multiple access (NOMA) in 5G systems: Potentials and
challenges,” _IEEE Commun. Surv. Tut._, vol. 19, no. 2, pp. 721–742, 2nd
Quart. 2017.

[284] M. T. P. Le, G. C. Ferrante, G. Caso, L. De Nardis, and M.-G. Di
Benedetto, “On information-theoretic limits of code-domain NOMA for
5G,” _IET Commun._, vol. 12, no. 15, pp. 1864–1871, Sep. 2018.

[285] B. Clerckx, Y. Mao, R. Schober, and H. V. Poor, “Rate-splitting unifying
SDMA, OMA, NOMA, and multicasting in MISO broadcast channel:
A simple two-user rate analysis,” _IEEE Wireless Commun. Lett._, vol. 9,
no. 3, pp. 349–353, Mar. 2020.

[286] L. Zhu, Z. Xiao, X.-G. Xia, and D. O. Wu, “Millimeter-wave communications with non-orthogonal multiple access for B5G/6G,” _IEEE Access_,
vol. 7, pp. 116123–116132, Aug. 2019.

[287] H. Marshoud, V. M. Kapinas, G. K. Karagiannidis, and S. Muhaidat, “Non-orthogonal multiple access for visible light communications,”
_IEEE Photon. Technol. Lett._, vol. 28, no. 1, pp. 51–54, Jan. 2016.

[288] Y. Al-Eryani and E. Hossain, “The D-OMA method for massive multiple access in 6G: performance, security, and challenges,” IEEE Vehic.
Technol. Mag., vol. 14, pp. 92–99, Sept. 2019.

[289] C. Chen, W.-D. Zhong, H. Yang, P. Du, and Y. Yang, “Flexible-rate
SIC-free NOMA for downlink VLC based on constellation partitioning
coding,” _IEEE Wireless Commun. Lett._, vol. 8, no. 2, pp. 568–571, Apr.
2019.

[290] E. Torkildson, U. Madhow, and M. Rodwell, “Indoor millimeter wave
MIMO: Feasibility and performance,” _IEEE Trans. Wireless Commun._,
vol. 10, no. 12, pp. 4150–4160, Dec. 2011.

[291] H. Do, N. Lee, and A. Lozano, “Reconfigurable ULAs for line-of-sight
[MIMO transmission,” 2020. [Online]. Available: https://arxiv.org/pdf/](https://arxiv.org/pdf/2004.12039.pdf)
[2004.12039.pdf](https://arxiv.org/pdf/2004.12039.pdf)

[292] H. Sarieddeen, M.-S. Alouini, and T. Y. Al-Naffouri, “An overview
of signal processing techniques for terahertz communications,” 2020.

[[Online]. Available: https://arxiv.org/pdf/2005.13176.pdf](https://arxiv.org/pdf/2005.13176.pdf)

[293] P. Wang, Y. Li, X. Yuan, L. Song, and B. Vucetic, “Tens of gigabits wireless communications over E-Band LoS MIMO channels with uniform

linear antenna arrays,” _IEEE Trans. Wireless Commun._, vol. 13, no. 7, pp.
3791–3805, Jul. 2014.



46 VOLUME 4, 2021


N. Rajatheva _et al._ : Scoring the Terabit/s Goal: Broadband Connectivity in 6G


[294] A. Faisal, H. Sarieddeen, H. Dahrouj, T. Y. Al-Naffouri, and M.-S.
Alouini, “Ultra-massive MIMO systems at terahertz bands: Prospects
[and challenges,” 2019. [Online]. Available: https://arxiv.org/pdf/1902.](https://arxiv.org/pdf/1902.11090.pdf)
[11090.pdf](https://arxiv.org/pdf/1902.11090.pdf)

[295] Y. Wang, J. Liang, S. Handagala, A. Madanayake, and S. Mandal, “ _δ −_ _σ_
noise-shaping in 2-D space-time for wideband antenna array receivers,”
_IEEE Trans. Circuits Syst. I_, vol. 66, no. 2, pp. 569–582, 2019.

[296] T. S. Rappaport _et al._, “Wireless communications and applications above
100 GHz: Opportunities and challenges for 6G and beyond,” _IEEE_
_Access_, vol. 7, pp. 78 729–78 757, 2019.

[297] P. Wang, Y. Li, Y. Peng, S. C. Liew, and B. Vucetic, “Non-uniform linear
antenna array design and optimization for millimeter-wave communications,” _IEEE Trans. Wireless Commun._, vol. 15, no. 11, pp. 7343–7356,
Nov. 2016.

[298] X. Song, W. Rave, N. Babu, S. Majhi, and G. Fettweis, “Two-level spatial
multiplexing using hybrid beamforming for millimeter-wave backhaul,”
_IEEE Trans. Wireless Commun._, vol. 17, no. 7, pp. 4830–4844, 2018.

[299] O. Simeone, “A very brief introduction to machine learning with applications to communication systems,” _IEEE Trans. Cogn. Commun. Netw._,
vol. 4, no. 4, pp. 648–664, Dec. 2018.

[300] T. O’Shea and J. Hoydis, “An introduction to deep learning for the
physical layer,” _IEEE Trans. Cogn. Commun. Netw._, vol. 3, no. 4, pp.
563–575, Dec. 2017.

[301] S. Dörner, S. Cammerer, J. Hoydis, and S. t. Brink, “Deep learning based
communication over the air,” _IEEE J. Sel. Topics Signal Process._, vol. 12,
no. 1, pp. 132–143, 2018.

[302] T. J. O’Shea, T. Erpek, and T. C. Clancy, “Physical layer deep learning
of encodings for the MIMO fading channel,” in _Proc. Allerton Conf._
_Commun., Contr., and Comput._, 2017.

[303] T. Erpek, T. J. O’Shea, and T. C. Clancy, “Learning a physical layer
scheme for the MIMO interference channel,” in _Proc. IEEE Int. Conf._
_Commun. (ICC)_, 2018.

[304] B. Zhu, J. Wang, L. He, and J. Song, “Joint transceiver optimization for
wireless communication PHY using neural network,” _IEEE J. Sel. Areas_
_Commun._, vol. 37, no. 6, pp. 1364–1373, 2019.

[305] H. Ye, G. Y. Li, B. F. Juang, and K. Sivanesan, “Channel agnostic endto-end learning based communication systems with conditional GAN,” in
_Proc. IEEE Global Commun. Conf. (GLOBECOM)_, 2018.

[306] T. J. O’Shea, T. Roy, N. West, and B. C. Hilburn, “Physical layer
communications system design over-the-air using adversarial networks,”
in _Proc. Eur. Signal Process. Conf. (EUSIPCO)_, 2018, pp. 529–532.

[307] F. A. Aoudia and J. Hoydis, “Model-free training of end-to-end communication systems,” _IEEE J. Sel. Areas Commun._, vol. 37, no. 11, pp.
2503–2516, Nov. 2019.

[308] N. Ye, X. Li, H. Yu, L. Zhao, W. Liu, and X. Hou, “DeepNOMA: A
unified framework for NOMA using deep multi-task learning,” _IEEE_
_Trans. Wireless Commun._, vol. 19, no. 4, pp. 2208–2225, 2020.

[309] R. N. S. Rajapaksha, “Potential deep learning approaches for the physical
layer,” Master’s thesis, University of Oulu, Finland, 2019. [Online].
[Available: http://urn.fi/URN:NBN:fi:oulu-201908142760](http://urn.fi/URN:NBN:fi:oulu-201908142760)

[310] H. Ye, G. Y. Li, and B. Juang, “Power of deep learning for channel estimation and signal detection in OFDM systems,” _IEEE Wireless Commun._
_Lett._, vol. 7, no. 1, pp. 114–117, Feb. 2018.

[311] Ö. T. Demir and E. Björnson, “Channel estimation in massive MIMO
under hardware non-linearities: Bayesian methods versus deep learning,”
_IEEE Open J. Commun. Soc._, vol. 1, pp. 109–124, 2020.

[312] J. Liu, K. Mei, X. Zhang, D. Ma, and J. Wei, “Online extreme learning
machine-based channel estimation and equalization for OFDM systems,”
_IEEE Comm. Lett._, vol. 23, no. 7, pp. 1276–1279, July 2019.

[313] M. Eisen, C. Zhang, L. F. O. Chamon, D. D. Lee, and A. Ribeiro,
“Learning optimal resource allocations in wireless systems,” _IEEE Trans._
_Signal Process._, vol. 67, no. 10, pp. 2775–2790, May 2019.

[314] F. B. Mismar, B. L. Evans, and A. Alkhateeb, “Deep reinforcement learning for 5G networks: Joint beamforming, power control, and interference
coordination,” _IEEE Trans. Commun._, vol. 68, no. 3, pp. 1581–1592,
2020.

[315] A. Alkhateeb, S. Alex, P. Varkey, Y. Li, Q. Qu, and D. Tujkovic, “Deep
learning coordinated beamforming for highly-mobile millimeter wave
systems,” _IEEE Access_, vol. 6, pp. 37 328–37 348, 2018.

[316] A. M. Elbir, “CNN-based precoder and combiner design in mmWave
MIMO systems,” _IEEE Communications Letters_, vol. 23, no. 7, pp. 1240–
1243, 2019.

[317] A. M. Elbir and A. K. Papazafeiropoulos, “Hybrid precoding for multiuser millimeter wave massive MIMO systems: A deep learning ap


proach,” _IEEE Trans. Veh. Technol._, vol. 69, no. 1, pp. 552–563, Jan.
2020.

[318] A. M. Elbir and K. V. Mishra, “Joint antenna selection and hybrid beamformer design using unquantized and quantized deep learning networks,”
_IEEE Trans. Wireless Commun._, vol. 19, no. 3, pp. 1677–1688, 2020.

[319] L. Sanguinetti, A. Zappone, and M. Debbah, “Deep learning power
allocation in massive MIMO,” in _Proc. Asilomar Conf. Signals, Syst., and_
_Comput. (ASILOMAR)_, 2018, pp. 1257–1261.

[320] T. V. Chien, T. N. Canh, E. Björnson, and E. G. Larsson, “Power control
in cellular massive MIMO with varying user activity: A deep learning
[solution,” 2019. [Online]. Available: https://arxiv.org/pdf/1901.03620.](https://arxiv.org/pdf/1901.03620.pdf)
[pdf](https://arxiv.org/pdf/1901.03620.pdf)

[321] Y. S. Nasir and D. Guo, “Multi-agent deep reinforcement learning for
dynamic power allocation in wireless networks,” _IEEE J. Sel. Areas_
_Commun._, vol. 37, no. 10, pp. 2239–2250, 2019.

[322] C. D’Andrea, A. Zappone, S. Buzzi, and M. Debbah, “Uplink power
control in cell-free massive MIMO via deep learning,” in _Int. Workshop_
_Comput. Adv. in Multi-Sensor Adaptive Process. (CAMSAP)_, 2019.

[323] S. Ali, W. Saad, and D. S. (eds.), “6G White paper on machine learning
in wireless communication networks,” June 2020. [Online]. Available:
[http://urn.fi/urn:isbn:9789526226736](http://urn.fi/urn:isbn:9789526226736)

[324] M. A. Maddah-Ali and U. Niesen, “Fundamental limits of caching,” _IEEE_
_Trans. Inf. Theory_, vol. 60, no. 5, pp. 2856–2867, May 2014.

[325] S. P. Shariatpanahi, S. A. Motahari, and B. H. Khalaj, “Multi-server
coded caching,” _IEEE Trans. Inf. Theory_, vol. 62, no. 12, pp. 7253–7271,
Dec. 2016.

[326] S. P. Shariatpanahi, G. Caire, and B. H. Khalaj, “Physical-layer schemes
for wireless coded caching,” _IEEE Trans. Inf. Theory_, vol. 65, no. 5, pp.
2792–2807, May 2019.

[327] A. Tölli, S. P. Shariatpanahi, J. Kaleva, and B. Khalaj, “Multicast beamformer design for coded caching,” in _Proc. IEEE Int. Symp. Inf. Theory_
_(ISIT)_, 2018.

[328] A. Tölli, S. P. Shariatpanahi, J. Kaleva, and B. Khalaj, “Multi-antenna
interference management for coded caching,” _IEEE Trans. Wireless Com-_
_mun._, vol. 19, no. 3, pp. 2091–2106, Mar. 2020.

[329] E. Lampiris and P. Elia, “Adding transmitters dramatically boosts codedcaching gains for finite file sizes,” _IEEE J. Sel. Areas Commun._, vol. 36,
no. 6, pp. 1176–1188, June 2018.

[330] M. Salehi, A. Tölli, and S. P. Shariatpanahi, “A multi-antenna coded
caching scheme with linear subpacketization,” in _Proc. IEEE Int. Conf._
_Commun. (ICC)_, May 2019.

[331] M. Salehi, A. Tolli, S. P. Shariatpanahi, and J. Kaleva, “Subpacketizationrate trade-off in multi-antenna coded caching,” in _Proc. IEEE Global_
_Commun. Conf. (GLOBECOM)_, 2019.

[332] M. Salehi, A. Tölli, and S. P. Shariatpanahi, “SubpacketizationBeamformer Interaction in Multi-Antenna Coded Caching,” in _Proc. 6G_
_Wireless Summit (6G SUMMIT)_, 2020.

[333] A. Destounis, M. Kobayashi, G. Paschos, and A. Ghorbel, “Alpha fair
coded caching,” in _Proc. Int. Symp. Model. and Optim. in Mobile, Ad_
_Hoc and Wireless Netw. (WiOpt)_, June 2017.

[334] H. Yang and T. L. Marzetta, “Energy efficiency of massive mimo: Cellfree vs. cellular,” in _Proc. IEEE Veh. Technol. Conf. (VTC-Spring)_, 2018.

[335] T. Taheri, R. Nilsson, and J. van de Beek, “The Potential of MassiveMIMO on TV Towers for Cellular Coverage Extension,” _Wireless_
_Communications and Mobile Computing_, vol. 2021, 2021. [Online].
[Available: https://doi.org/10.1155/2021/8164367](https://doi.org/10.1155/2021/8164367)



VOLUME 4, 2021 47


NANDANA RAJATHEVA is currently a Professor with the Centre for Wireless Communications

(CWC), University of Oulu, Finland. He is a Senior Member, IEEE and received the B.Sc. (Hons.)
degree in electronics and telecommunication engineering from the University of Moratuwa, Sri
Lanka, in 1987, ranking first in the graduating
class, and the M.Sc. and Ph.D. degrees from the
University of Manitoba, Winnipeg, MB, Canada,
in 1991 and 1995, respectively. He was a Canadian
Commonwealth Scholar during the graduate studies in Manitoba. He held
Professor/Associate Professor positions at the University of Moratuwa and
the Asian Institute of Technology (AIT), Thailand, from 1995 to 2010. He
has co-authored more than 200 refereed papers published in journals and
in conference proceedings. His research interests include physical layer
in beyond 5G, machine learning for PHY & MAC, sensing for factory
automation and channel coding. He is currently leading the AI-driven Air
Interface design task in Hexa-X EU Project.


ITALO ATZENI received the PhD degree
(Hons.) in signal theory and communications
from the Polytechnic University of Catalonia–BarcelonaTech in 2014. He is Senior Research

Fellow and Adjunct Professor at the Centre for
Wireless Communications, University of Oulu.
He was with the Mathematical and Algorithmic Sciences Laboratory, Paris Research Center,
Huawei Technologies from 2014 to 2017 and
with the Communication Systems Department,
EURECOM from 2017 to 2018. His primary research interests are in
communication and information theory, statistical signal processing, and
convex and distributed optimization theory. He received the Best Paper
Award in the Wireless Communications Symposium at IEEE ICC 2019. He
was recently granted the MSCA-IF for the project DELIGHT.


SIMON BICAÏS received the M.Sc. in telecom
munications (2017), from the National Institute of
Applied Sciences of Lyon (INSA Lyon), France
and a PhD in 2020 from Grenoble University,
France. Signal processing, wireless communications and machine learning are his current research
interests. He was involved in the BRAVE project
about Beyond 5G wireless communications in the
sub-TeraHertz bands. He is the main inventor of 4

patents.


EMIL BJÖRNSON is a Professor at the KTH

Royal Institute of Technology, Sweden, and an
Associate Professor at Linköping University, Sweden. He has authored the textbooks _Optimal Re-_
_source Allocation in Coordinated Multi-Cell Sys-_
_tems_ (2013), _Massive MIMO Networks: Spec-_
_tral, Energy, and Hardware Efficiency_ (2017), and
_Foundations of User-Centric Cell-Free Massive_
_MIMO_ (2021). He has received the 2014 Outstanding Young Researcher Award from IEEE ComSoc
EMEA, the 2016 Best Ph.D. Award from EURASIP, the 2018 IEEE Marconi
Prize Paper Award in Wireless Communications, the 2019 EURASIP Early
Career Award, the 2019 IEEE Communications Society Fred W. Ellersick
Prize, the 2019 IEEE Signal Processing Magazine Best Column Award, the
2020 Pierre-Simon Laplace Early Career Technical Achievement Award, and
the 2020 CTTC Early Achievement Award. He also co-authored papers that
received Best Paper Awards at WCSP 2009, the IEEE CAMSAP 2011, the
IEEE WCNC 2014, the IEEE ICC 2015, WCSP 2017, and the IEEE SAM

2014.



N. Rajatheva _et al._ : Scoring the Terabit/s Goal: Broadband Connectivity in 6G


ANDRÉ BOURDOUX received the M.Sc. degree
in electrical engineering in 1982 from the Université Catholique de Louvain-la-Neuve, Belgium.
He joined IMEC in 1998 and is Principal Member
of Technical Staff in the IoT Research Group of
IMEC. He is a system level and signal processing expert for both the mm-wave wireless communications and radar teams. He has more than

15 years of research experience in radar systems
and 15 years of research experience in broadband
wireless communications. He holds several patents in these fields. He is the
author and co-author of over 160 publications in books and peer-reviewed
journals and conferences. His research interests are in the field of advanced
architectures, signal processing and machine learning for wireless physical
layer and high-resolution 3D/4D radars.


STEFANO BUZZI is currently a Professor at the
Department of Electrical and Information Engineering at the University of Cassino and Southern
Latium. He is a former editor of the IEEE Commu
nications Letters and of the IEEE Signal Processing Letters, while is currently serving as Associate
Editor of the IEEE Transactions on Wireless Com
munications. He has co-authored more than 150

technical papers published in international journals and in international conference proceedings.
His research interests are focused on the PHY and MAC layer of wireless
networks.


CARMEN D’ANDREA was born in Caserta, Italy
on 16 July 1991. She received the B.S. and M.S.
degrees, both with honors, in Telecommunications
Engineering from the University of Cassino and
Lazio Meridionale in 2013 and 2015, respectively.
In 2017, she was a Visiting Ph.D. student with
the Wireless Communications (WiCom) Research
Group in the Department of Information and Communication Technologies at Universitat Pompeu
Fabra in Barcelona, Spain. In 2019, she received
the Ph.D. degree with the highest marks in Electrical and Information
Engineering from the University of Cassino and Lazio Meridionale where
she is currently a post-doc researcher. Her research interests are focused on
wireless communication and signal processing, with current emphasis on
mmWave communications and massive MIMO systems, in both colocated
and distributed setups.


JEAN-BAPTISTE DORÉ received his MS de
gree in 2004 from the Institut National des Sciences Appliquées (INSA) Rennes, France and his
PhD in 2007. He joined NXP semiconductors as
a signal processing architect. Since 2009 he has
been with CEA-Leti in Grenoble, France as a
research engineer and program manager. His main
research topics are signal processing (waveform
optimization and channel coding), hardware architecture optimizations (FPGA, ASIC), PHY and
MAC layers for wireless networks. Jean-Baptiste Doré has published 50+
papers in international conference proceedings and book chapters, received
2 best papers award (ICC2017, WPNC2018). He has also been involved in
standardization group (IEEE1900.7) and is the main inventor of more than
30 patents.



48 VOLUME 4, 2021


N. Rajatheva _et al._ : Scoring the Terabit/s Goal: Broadband Connectivity in 6G


SERHAT ERKUCUK is a Professor in the De
partment of Electrical-Electronics Engineering at
Kadir Has University, Turkey. He received his
Ph.D. degree in engineering science from Simon Fraser University and held an NSERC postdoctoral fellowship at the University of British
Columbia. He has co-authored more than 60 papers published in international journals and conference proceedings in the areas of PHY and MAC
layer design of wireless communication systems,
and has been granted 2 US patents. He is a Marie Curie Fellow and a
recipient of Governor General’s Gold Medal.


MANUEL FUENTES received the Ph.D. degree
in Telecommunication Engineering from Universitat Politecnica de Valencia (UPV), in 2017.
In 2012-2017 and 2018-2020, he was with the

the Institute of Telecommunications and Multi
media Applications (iTEAM) at UPV. He was
also a guest researcher at the Vienna University
of Technology, Austria, in 2016. In 2017-2018,
Dr. Fuentes worked with the Samsung Electronics
R&D UK team as a 5G research engineer. He
contributed actively to the ATSC 3.0 standardization process and participated in the IMT-2020 Evaluation Group of 5G PPP. He is co-author of
20+ international IEEE journal and conference papers. He also received the
IEEE BTS Scott Helt Award to the best paper of the IEEE Transactions on
Broadcasting in 2019. In September 2020, Dr. Fuentes joined Fivecomm as
an R&D Manager. He is currently leading a team to work in several Horizon
2020 European 5G/IoT projects. His main areas of interest include physical
layer and radio procedures, 5G integration, development and demonstration,
as well as Beyond-5G and 6G communications.


KE GUAN is a Full Professor in State Key Laboratory of Rail Traffic Control and Safety, Beijing
Jiaotong University (BJTU). In 2015, he has been
awarded a Humboldt Research Fellowship. He
has authored/co-authored more than 240 journal
and conference papers, receiving eight Best Paper
Awards, including IEEE vehicular technology society 2019 Neal Shepherd memorial best propagation paper award. His current research interests
include measurement and modeling of wireless
propagation channels for various applications in the era of 5G and beyond.
He is an Editor of the IEEE Vehicular Technology Magazine, the IEEE
ACCESS, and the IET Microwave, Antenna and Propagation. He is the
contact person of BJTU in 3GPP and a member of the IC1004 and CA15104
initiatives.



XIAOJING HUANG received the B.Eng.,
M.Eng., and Ph.D. degrees in electronic engineering from Shanghai Jiao Tong University, Shanghai, China, in 1983, 1986, and 1989, respectively.
He was a Principal Research Engineer with the
Motorola Australian Research Center, Botany,
NSW, Australia, from 1998 to 2003, and an Associate professor with the University of Wollongong, Wollongong, NSW, Australia, from 2004 to
2008. He had been a Principal Research Scientist
with the Commonwealth Scientific and Industrial Research Organisation
(CSIRO), Sydney, NSW, Australia, and the Project Leader of the CSIRO
Microwave and mm-Wave Backhaul projects since 2009. He is currently a
Professor of Information and Communications Technology with the School
of Electrical and Data Engineering and the Program Leader for Mobile
Sensing and Communications with the Global Big Data Technologies
Center, University of Technology Sydney (UTS), Sydney, NSW, Australia.


JARI HULKKONEN graduated in 1999 (M.Sc.EE)
from University of Oulu, Finland. He has been
working in Nokia since 1996. He started his career in GSM/EDGE research and standardization

projects. Since 2006 he has been leading radio
systems research in Oulu. Currently Jari is Radio
Research Department Head in Nokia Bell Labs
Oulu with focus on 5G New Radio evolution. He

has more than 30 granted patents/patent applications in 2G-5G technologies.


JOSEP MIQUEL JORNET received the Ph.D.

degree in Electrical and Computer Engineering
(ECE) from the Georgia Institute of Technology
in 2013. Between 2013 and 2019, he was with the
Department of Electrical Engineering at University at Buffalo. Since August 2019, he has been an
Associate Professor in the Department of ECE at
Northeastern University. His research interests are
in terahertz communications and wireless nano
bio-communication networks. He has co-authored
more than 120 peer-reviewed scientific publications, one book, and has been
granted 3 US patents, and is serving as the lead PI on multiple grants from
U.S. federal agencies.


MARCOS KATZ is a Professor at the Centre

for Wireless Communications, University of Oulu,
Finland, since Dec. 2009. He received the B.S.
degree in Electrical Engineering from Universidad
Nacional de Tucumán, Argentina in 1987, and the
M.S. and Dr. Tech. degrees in Electrical Engineering from University of Oulu, Finland, in 1995 and
2002, respectively. He worked in different positions at Nokia, Finland between 1987 and 2001.
In years 2003–2005 Dr. Katz was the Principal
Engineer at Samsung Electronics, Advanced Research Lab., Telecommunications R/D Center, Suwon, Korea. From 2006 to 2009 he worked as a Chief
Research Scientist at VTT, the Technical Research Centre of Finland.



PLACE

PHOTO

HERE



YUZHOU HU is a senior algorithm engineer with
ZTE Corporation. He has served as 3GPP delegate and researcher in the field of non-orthogonal
multiple access (NOMA), 2-step random access
(2SR), 5G V2X since joining ZTE in 2017. He
has filed over 50 patents. He received a BSc and
MSc(Summa Cum Laude) in Mathematics and
Electronics Engineering from Beihang University,
China.



VOLUME 4, 2021 49


BEHROOZ MAKKI [M’19, SM’19] received his
PhD degree in Communication Engineering from
Chalmers University of Technology, Gothenburg,
Sweden. In 2013-2017, he was a Postdoc researcher at Chalmers University. Currently, he
works as Senior Researcher in Ericsson Research,
Gothenburg, Sweden.
Behrooz is the recipient of the VR Research
Link grant, Sweden, 2014, the Ericsson’s Research
grant, Sweden, 2013, 2014 and 2015, the ICT
SEED grant, Sweden, 2017, as well as the Wallenbergs research grant,
Sweden, 2018. Also, Behrooz is the recipient of the IEEE best reviewer
award, IEEE Transactions on Wireless Communications, 2018. Currently,
he works as an Editor in IEEE Wireless Communications Letters, IEEE
Communications Letters, the journal of Communications and Information
Networks, as well as the Associate Editor in Frontiers in Communications
and Networks. He was a member of European Commission projects “mmWave based Mobile Radio Access Network for 5G Integrated Communications” and “ARTIST4G” as well as various national and international

research collaborations. His current research interests include integrated
access and backhaul, Green communications, millimeter wave communications, finite block-length analysis and backhauling. He has co-authored 63
journal papers, 46 conference papers and 60 patent applications.


RICKARD NILSSON received the Ph.D. degree
from Luleå University of Technology (LTU), Sweden. With Telia Research AB, Sweden, and Stanford University, USA, he introduced a new flexible broadband access method for VDSL and con
tributed to its standardization. For seven years he
was a senior researcher at the Telecommunications

Research Center Vienna, Austria, and lectured at
the Technical University. Since 2010 he is with
LTU researching wireless connectivity, lecturing
signal processing and communications, and cooperating with industry.


ERDAL PANAYIRCI received the Ph.D. degree
in electrical engineering from Michigan State University, Michigan, USA. He is currently a professor of electrical engineering in the Electrical and
Electronics Engineering Department at Kadir Has
University, Istanbul, Turkey and Visiting Research
Collaborator at the Department of Electrical Engineering, Princeton University, USA. He has published extensively in leading scientific journals and
international conference and co-authored the book

Principles of Integrated Maritime Surveillance Systems (Kluwer Academic,
2000). His research interests are advanced signal processing techniques
and their applications to wireless electrical, underwater and optical communications. Prof. Panayirci was an Editor for the IEEE transactions on
communications and served and is currently serving as a Member of IEEE
Fellow Committee during 2005–2008 and 2019– 2021, respectively. He is
an IEEE Life Fellow.


KHALED RABIE received the Ph.D. degree in
Electrical and Electronic Engineering from the
University of Manchester, UK, in 2015. He is currently an Assistant Professor with the department
of Engineering at the Manchester Metropolitan
University, UK. His primary research focuses on
various aspects of the next-generation wireless
communication systems. He serves as an Editor
for IEEE COMMUNICATIONS LETTERS, an
Associate Editor for IEEE ACCESS, and an Area
Editor for PHYSICAL COMMUNICATIONS. He received the Best Paper
Award at the IEEE ISPLC 2015 as well as the IEEE ACCESS Editor of the

month award for August 2019. Khaled is also a Fellow of the U.K. Higher
Education Academy.



N. Rajatheva _et al._ : Scoring the Terabit/s Goal: Broadband Connectivity in 6G


NUWANTHIKA RAJAPAKSHA is currently a
doctoral student at the Centre for Wireless Com
munications, University of Oulu, Finland. She received the B.Sc. (Hons.) degree in electronics and
telecommunication engineering from University
of Moratuwa, Sri Lanka, in 2014, and the M.Sc.
degree in wireless communications engineering
from University of Oulu, in 2019. From 2014 to
2018 she was a research engineer at Synergen
Technology Labs, Sri Lanka, where she was involved in biomedical signal processing, machine learning-based algorithm
development, and US FDA regulatory process for wearable medical device
design. Her main research interests are signal processing and machine
learning applications in PHY/MAC layers.


MOHAMMADJAVAD SALEHI received his PhD

in Electrical Engineering from Sharif University
of Technology, Tehran, Iran, in 2018. His career
in Iran also includes 9 years of work experience
in the IT industry, as a developer, manager and
start-up founder. Since 2019, he is a postdoctoral
researcher at University of Oulu, Finland, where
he works on various aspects of practical implementation of coded caching techniques in future
wireless networks. Specifically, his research now
includes designing coded caching schemes with reduced subpacketization,
performance analysis of multi-antenna coded caching at finite-SNR communication regime, and energy efficiency of practical coded caching setups.


HADI SARIEDDEEN received the B.E. degree
(summa cum laude; first in graduating class) in
computer and communications engineering from
Notre Dame University-Louaize (NDU), Lebanon,
in 2013, and the Ph.D. degree in electrical and
computer engineering from the American University of Beirut (AUB), Lebanon, in 2018. He is
currently a postdoctoral research fellow at King
Abdullah University of Science and Technology
(KAUST), Thuwal, Saudi Arabia. His research
interests are in the areas of communication theory and signal processing for
wireless communications.


SHAHRIAR SHAHABUDDIN received his MSc

and PhD from University of Oulu, Finland in
2012 and 2019 respectively. During Spring 2015,
he worked as a Visiting Researcher at Computer
Systems Laboratory, Cornell University, USA.
Shahriar received distinction in MSc and several

scholarships and grants such as Nokia Foundation
Scholarship, University of Oulu Scholarship Foundation Grant, Tauno Tönning Foundation Grant
during his PhD. Shahriar’s research interest includes VLSI signal processing, massive MIMO systems and physical layer
security. Since 2017, Shahriar has been with Nokia, Finland as a SoC
Specialist.



50 VOLUME 4, 2021


N. Rajatheva _et al._ : Scoring the Terabit/s Goal: Broadband Connectivity in 6G


TOMMY SVENSSON is Full Professor at

Chalmers University of Technology in Gothenburg, Sweden, leading Wireless Systems. He has a
PhD in Information theory from Chalmers in 2003,
and has worked at Ericsson AB with core, radio
access, and microwave networks. He has been
deeply involved in European research towards 4G,
5G and currently towards 6G within the HexaX, RISE-6G and SEMANTIC EU projects, on access, backhaul/ fronthaul, C-V2X and satellite networks, with a focus on physical and MAC layer. He has coauthored 5 books,
93 journal and 129 conference papers, and 53 EU projects deliverables. He
is chair of the IEEE Sweden VT/COM/IT chapter, founding editorial board
member of IEEE JSAC Series on Machine Learning in Communications and
Networks, been editor of IEEE TWC, WCL, guest editor of top journals,
organizer of tutorials/ workshops at top IEEE conferences, and coordinator
of the Communication Engineering MSc program at Chalmers.


OSKARI TERVO is currently working as a Senior
Standardization Research Specialist in Nokia Bell
Labs, Oulu, where he joined in 2018. He received
a doctoral degree from Centre for Wireless Communications (CWC), University of Oulu, Finland,
in 2018 with distinction. In 2014 and 2016, he was
a Visiting Researcher with Kyung Hee University,
Seoul, Korea, and the Interdisciplinary Centre for
Security, Reliability and Trust (SnT), University
of Luxembourg, Luxembourg, respectively. He has
received the Best Reviewer Award from IEEE Transactions on Wireless

Communications in 2017 and 2018.


ANTTI TÖLLI is an Associate Professor with

the Centre for Wireless Communications (CWC),
University of Oulu. He received the Dr.Sc. (Tech.)
degree in electrical engineering from the University of Oulu, Oulu, Finland, in 2008. From 1998 to
2003, he worked at Nokia Networks as a Research
Engineer and Project Manager both in Finland
and Spain. In May 2014, he was granted a five
year (2014-2019) Academy Research Fellow post
by the Academy of Finland. During the academic
year 2015-2016, he visited at EURECOM, Sophia Antipolis, France, while
from August 2018 till June 2019 he was visiting at the University of
California Santa Barbara, USA. He is currently serving as an Associate
Editor for IEEE Transactions on Signal Processing.


QINGQING WU is an assistant professor in the
University of Macau, Macau, China. His current
research interests include IRS-enabled 6G, UAV
communications, and green communications. He
has published over 60 IEEE top-tier journal and
conference papers, which have attracted more than
3400 Google Scholar citations. He received the
Best Ph.D. Thesis Award of China Institute of

Communications in 2017 and the IEEE WCSP

Best Paper Award in 2015. He serves as an Associate Editor for IEEE CL, the Guest Editor for IEEE OJVT on “6G Intelligent
Communications", and the Leading Guest Editor for IEEE JSAC on “UAV
Communications in 5G and Beyond Networks".



WEN XU received the Dr.-Ing. degree from Technical University of Munich, Germany, in 1996.
From 1995 to 2006, he was with Siemens AG,
Munich, where he was head of the Algorithms and
Standardization Lab. From 2007 to 2014, he was
with Infineon Technologies AG ( _later_ Intel Mobile
Communications GmbH), Germany. In 2014, he
joined Huawei Technologies Duesseldorf GmbH

            - Munich Research Center, where he is leading
the Radio Access Technologies Dept. His research
interests include signal processing, source/channel coding, and wireless
communication systems. He has 100+ peer-reviewed papers published and
numerous patents granted.



VOLUME 4, 2021 51


