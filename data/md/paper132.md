Temporal information processing on noisy quantum computers


Jiayin Chen, [1, 2] Hendra I. Nurdin, [1,][ ∗] and Naoki Yamamoto [2, 3]

1 School of Electrical Engineering and Telecommunications,
The University of New South Wales (UNSW), Sydney NSW 2052, Australia.
2 Quantum Computing Center, Keio University, Hiyoshi 3-14-1, Kohoku, Yokohama 223-8522, Japan.
3 Department of Applied Physics and Physico-Informatics,
Keio University, Hiyoshi 3-14-1, Kohoku, Yokohama 223-8522, Japan.


The combination of machine learning and quantum computing has emerged as a promising approach for addressing previously untenable problems. Reservoir computing is an efficient learning
paradigm that utilizes nonlinear dynamical systems for temporal information processing, i.e., processing of input sequences to produce output sequences. Here we propose quantum reservoir computing that harnesses complex dissipative quantum dynamics. Our class of quantum reservoirs is
universal, in that any nonlinear fading memory map can be approximated arbitrarily closely and uniformly over all inputs by a quantum reservoir from this class. We describe a subclass of the universal
class that is readily implementable using quantum gates native to current noisy gate-model quantum computers. Proof-of-principle experiments on remotely accessed cloud-based superconducting
quantum computers demonstrate that small and noisy quantum reservoirs can tackle high-order
nonlinear temporal tasks. Our theoretical and experimental results pave the path for attractive
temporal processing applications of near-term gate-model quantum computers of increasing fidelity
but without quantum error correction, signifying the potential of these devices for wider applications
including neural modeling, speech recognition and natural language processing, going beyond static
classification and regression tasks.



I. INTRODUCTION


The ingenious use of quantum effects has led to a significant number of quantum machine learning algorithms
that offer computational speed-ups [1, 2]. While awaiting
the demonstration of these quantum algorithms on fullfledge quantum computers equipped with quantum error correction, quantum computing has transitioned from
theoretical ideas to the noisy intermediate-scale quantum
(NISQ) technology era [3]. Hybrid quantum-classical algorithms using short-depth circuits are particularly suitable for implementation on NISQ devices. Many notable
experimental demonstrations of NISQ devices employ hybrid algorithms for data classification [4] and quantum
chemistry [5]. An on-going quest is to find interesting applications on quantum computers with increasingly lower
noise profile but not reaching a low enough threshold to
enable continuous quantum error correction.
Here we propose a hybrid quantum-classical algorithm
that utilizes dissipative quantum dynamics as reservoir
computers (RC) for temporal information processing on
gate-model NISQ quantum computers. The goal for temporal information processing tasks, such as speech processing and natural language processing [6, 7], is to learn
the relationship between input sequences and output sequences. The RC framework uses an arbitrary but fixed
dynamical system (in this case systems with dynamics
described by state-space difference equation), the “reservoir”, to map sequential inputs into its high-dimensional
state-space. Only a simple linear regression algorithm is
required to optimize the parameters of a readout function


∗ [h.nurdin@unsw.edu.au](mailto:h.nurdin@unsw.edu.au)



to approximate target outputs. The use of a simple linear readout has connections to the biological concept of
mixed selectivity, as demonstrated in monkeys [8]. The
attractiveness of the RC scheme is that naturally occurring dynamical systems (with some desired properties)
in physics and engineering can be exploited for temporal
information processing without precise tuning of its parameters, circumventing the expensive training cost in
alternative schemes such as recurrent neural networks
with tunable internal weights [9]. The ease of RC implementation has brought forward many successful hardware implementations of classical (i.e., non-quantum) RC
schemes [10–12]. A spintronic RC achieved state-of-theart performance on a spoken digit recognition task [13]
and a photonic RC demonstrated high-speed speech classification with a low error [14]. For theoretical developments, [15] derives an approximation error upper bound
for certain classical RCs on learning a class of inputoutput maps (not necessarily fading memory maps considered here). Information processing capacity of various
RC schemes has been investigated [16, 17]. See [18, 19]
for further interests and developments of RCs.


In this work, we employ dissipative quantum systems
as quantum reservoirs (QRs) to approximate nonlinear
input-output maps with fading memory. A map has fading memory if its outputs depend increasingly less on
inputs from earlier times. These maps are important
in a broad class of real-world problems including spoken digit recognition [13] and neural modeling [20]. The
use of quantum systems as QRs was initially proposed
in [21, 22] to harness disordered-ensemble quantum dynamics for temporal information processing. This QR
class is suitable for ensemble quantum systems and a
static (non-temporal) version of [21] was demonstrated


in NMR to approximate static maps [23]. However, it
remained an open problem to show this QR class has
the properties required for reservoir computing. Chen
and Nurdin [24] addressed this problem by demonstrating that a variation of the scheme proposed in [21, 22]
is universal for nonlinear fading memory maps, meaning that given any target nonlinear fading memory map,
there exists a member in the universal QR class whose
outputs approximate the target map’s outputs arbitrarily
closely and uniformly over the input sequences. This is a
quantum analogue of the universal function approximation property feed-forward neural networks enjoy [25, 26],
but for nonlinear fading memory mappings from input sequences to output sequences. The notion of universality
we adopt here was previously established for classical RC
schemes [20, 27, 28] and the Volterra series [29]. In particular, [28] proves this universality property for a form
of recurrent neural networks called echo-state networks

[30]. However, realizing these previous QR proposals in
the quantum gate-model remains challenging due to the
large number of quantum gates required to implement
the dynamics via Trotterization.

The contribution of this work is twofold. Firstly, we
propose a new class of QRs endowed with the fading
memory and universality properties that is not implemented by Ising Hamiltonians, circumventing the need
for Trotterization required in previous proposals. Secondly, we propose a realization of a subclass of the universal QR class on NISQ devices and present proof-ofprinciple experiments on remotely accessed IBM superconducting quantum processors [31], i.e., NISQ devices
not yet equipped with quantum error correction. The QR
dynamics in this subclass can be implemented using arbitrary but fixed quantum circuits, as long as they generate
non-trivial dynamics. This could be, for instance, quantum circuits that are classically intractable to simulate.
The quantum circuits can be of short lengths and can be
implemented using parametrized single-qubit and multiqubit quantum gates native to the quantum hardware,
without the need for precise tuning of their gate parameters. Our proof-of-principle experiments show that QRs
with a small number of qubits operating in a noisy environment can tackle complex nonlinear temporal tasks,
even under current hardware limitations and in the absence of readout and process error mitigation techniques.
This work serves as the first theoretical and experimental realization of applying near-term gate-model quantum
computers to nonlinear temporal information processing
tasks, opening an avenue for time series modeling and
signal processing applications of these devices.

The rest of this paper is organized as follows. In Sec. II
we introduce fading memory maps and describe two
temporal information processing tasks for these maps.
Sec. III introduces the RC framework and explains conditions for which a RC defines a fading memory map.
Sec. IV presents our QR proposal and the universality
result. We then propose a subclass of the universal class
suitable for implementation on current noisy gate-model



2


quantum computers. We conclude the section by discussing invariance properties of the universal class under certain hardware imperfections. Sec. V details two
hardware realizations of the aforementioned subclass of
the universal one and presents more efficient versions of
both schemes that could enable QR’s potential for more
scalable temporal processing on gate-model quantum devices. Sec. VI details our proof-of-principle experiments
performed on cloud-based IBM superconducting quantum devices. We provide concluding remarks in Sec. VII.
Detailed mathematical derivations and experimental settings are provided in the Appendix.


II. TEMPORAL INFORMATION PROCESSING


We consider a input-output (I/O) map M that maps
infinite input sequences u = {. . ., u −1, u 0, u 1, . . .} to infinite output sequences y = {. . ., y −1, y 0, y 1, . . .}, where
u l, y l ∈ R for l ∈ Z and y l = M (u) l is the output at time l. We write u| L:L ′ = {u L, . . ., u L ′ } and
y| L:L ′ = {y L, . . ., y L ′ } to denote the inputs and outputs
during time l = L, . . ., L [′] . In practice, such I/O maps
can be realized by convergent dynamical systems, that
is, systems that forget their initial condition (see Appendix A 1 for details). If such a dynamical system with
state x l is initialized at time l 0 at the state x l 0 and given
an input sequence {u l 0, u l 0 +1, . . .} and the system outputs the sequence {y l 0, y l 0 +1, . . .}, then it realizes an I/O
map M for any initial condition x l 0 as l 0 →−∞.
Two challenging temporal information processing
problems are posed to learn the I/O relationship given
by M based on the I/O pair u, y. The first is the multistep ahead prediction problem, in which we are given
inputs u| 1:L and the corresponding outputs y| 1:L . The
first L T < L input-output data pair (u| 1:L T, y| 1:L T ) is
the train data. In the sequel, we use the input-output
train data during l = 5, . . ., L T . The reason for this is to
remove the transient response in the data, see Sec. VI B
for a discussion. The goal is to use the train data to
optimize the parameters w of another I/O map M w,
so that the outputs y| L T +1:L = {y L T +1, . . . y L }, where
y l = M w (u) l, approximate the target outputs y| L T +1:L .
The second problem is the map emulation problem, that
is to optimize w of M w to emulate M using k = 1, . . ., K
different I/O train data pairs (u [k] | 1:L ′, y [k] | 1:L ′ ), so that
the total number of train data is KL [′] (we will again use
train data during l = 5, . . ., l = L [′] in Sec. VI B). When
given a previously unseen input u [K][+1] | 1:L [′], the task is for
y [K][+1] | 1:L ′ to approximate y [K][+1] | 1:L ′ .
If an I/O map M has fading memory, then its output at
time l [′] becomes increasingly less dependent on input samples u l from much earlier times l ≪ l [′] ; see Appendix A 2.
In this work, we approximate nonlinear fading memory
I/O maps using RCs implemented by quantum dynamical systems. We will introduce conditions for which a
reservoir dynamical system defines a fading memory I/O
map in the next section.


III. RESERVOIR COMPUTING


To approximate fading memory maps, RC exploits
nonlinear dynamical systems to project the input u l into
a reservoir state x l at time l. A RC is governed by a
dynamics f with state evolution x l = f (x l−1, u l ). The
dynamics of the reservoir can be arbitrary but fixed as
long as it satisfies some required properties, and never
requires training. We require the RC to satisfy the echostate property [30] or the convergence property [32], so
that the RC asymptotically forgets its initial condition.
The tunable parameters w appear in a readout function
h w, which combines the elements of x l into an output
y l = h w (x l ). For a sufficiently long input sequence
{u l 0, u l 0 +1, . . ., u 0 }, the effect of the RC’s initial condition can be washed-out. As discussed in Sec. II, as
l 0 →−∞, the combination of a convergent RC dynamics f and the readout function h w produces an I/O map
M (f,h w ) . After the washout, the readout parameters w
can be optimized using linear regression to minimize an
empirical mean squared-error between y 1:L T and y 1:L T .
As in previous works [20, 27–29], we consider M (f,h w )
that has the fading memory property.
Echo-state networks, one of the pioneering classical RC
schemes, have been numerically demonstrated to achieve
state-of-the-art performance in chaotic system modeling

[30]. Subsequent hardware realizations of RC proposals
exploit classical dynamical systems for real-time temporal processing tasks that demand less energy or computational memory [10–14]. These experiments also suggest
empirically that for certain tasks, such as spoken digit
recognition, the reservoir state dimension plays a role in
the RC’s task performance.


IV. UNIVERSAL QUANTUM RESERVOIR
COMPUTERS


We propose to use a QR, with a view towards possibly taking advantage of fast quantum dynamics and its
exponentially large state space. A QR consists of N noninteracting subsystems, each subsystem k has n k number
of qubits so that the QR has n = [�] [N] k=1 [n] [k] [ qubits. The]
QR density operator ρ l at time l evolves according to



3


for input 0 ≤ u l ≤ 1. Here, 0 < ǫ k ≤ 1, σ k is an arbitrary
but fixed density operator, and T 0 [(][k][)] and T 1 [(][k][)] are two
arbitrary but fixed completely positive trace-preserving
(CPTP) maps. Examples of such maps include some naturally occurring noisy quantum channels, such as dephasing or amplitude damping channels; see [33]. No precise
tuning or engineering of the CPTP maps T 0 [(][k][)], T 1 [(][k][)] is
required for the QR scheme and it should not generate
trivial dynamics (i.e., we should not choose T 0 [(][k][)] = T 1 [(][k][)] ).
They could potentially be classically intractable to simulate CPTP maps. The QR dynamics Eq. (1)–(2) is convergent, meaning that it will asymptotically forget its
initial condition; see Appendix A 1 for the proof. Given
inputs {u l 0, u l 0 +1, . . ., u 0 } and l 0 →−∞, the convergence property ensures the QR state ρ 0 evolves according
to Eqs. (1)–(2) is determined by {u l 0, u l 0 +1, . . ., u 0 } and
T 0 [(][k][)], T 1 [(][k][)], but not by its initial state ρ l 0 .


We obtain partial information about ρ l by measuring each qubit in the Pauli Z basis to obtain ⟨Z [(][i][)] ⟩ l =
Tr(ρ l Z [(][i][)] ) for i = 1, . . ., n, where Z [(][i][)] acts on qubit i. We
associate the readout function Eq. (3) to the QR dynamics Eq. (1). The readout function Eq. (3) is a multivariate
polynomial of degree R in the variables ⟨Z [(][i] [j] [)] ⟩ l . A simple
linear form (R = 1) is employed in our proof-of-principle
experiments in Sec. VI. The tunable readout parameters
r i1,...,r in
w = {w i 1,...,i n, w c } can be optimized via linear regression. Eqs. (1) and (3) define a QR implementing an I/O
map M (T,h w ) that depends on the QR dynamics T and
the readout function h w . We show in Appendix A 2 that
M (T,h w ) has the fading memory property. Now consider
the class M of I/O maps M (T,h w ) arising from differing
numbers of subsystems N, numbers of qubits n, QR dynamics T (u l ), readout parameters w and degree R of h w .
Our main result shows that the class M is universal for
approximating nonlinear fading memory maps.


Theorem 1 (Universality). Let K([0, 1]) be the set of
input sequences {u l } with 0 ≤ u l ≤ 1 for l ∈ Z. For any
nonlinear fading memory map M and any δ > 0, there
exists M (T,h w ) ∈M implemented by some QR such that
for all u ∈ K([0, 1]), sup l∈Z ��M (u) l − M (T,h w ) (u) l �� < δ.


We remark that universality is a property of the QR
class M and not of an individual member of M. The
universality proof employs the Stone-Weierstrass Theorem [34, Theorem 7.3.1], see Appendix A 3 for the proof.
Besides the universality property, our proposed universal QR class exhibits invariance properties under certain
hardware imperfections; see Sec. IV B below.



ρ l = T (u l )ρ l−1 =



N
� T [(][k][)] (u l )ρ [(] l− [k][)] 1 [,] (1)


k=1



and the k-th subsystem density operator ρ [(] l [k][)] undergoes
the evolution


T [(][k][)] (u l )ρ [(] l− [k][)] 1

(2)
= (1 − ǫ k ) u l T 0 [(][k][)] + (1 − u l )T 1 [(][k][)] ρ [(] l− [k][)] 1 [+][ ǫ] [k] [σ] [k] [,]
� �



n


    - · ·

�

i 1 =1



� w ri 1i1,...,i,...,r n in ⟨Z [(][i] [1] [)] ⟩ lr i1 - · · ⟨Z [(][i] [n] [)] ⟩ l [r] [in] + w c, (3)

r i1 +···+r in =d



n
�

i n =i n−1 +1



y l = h w (ρ l ) =



R
�


d=1


FIG. 1. Quantum circuit interpretation of the QR universal
subclass described in Sec. IV A. Here ρ [(] l− [k][)] 1 [and][ σ] [k] [ are two]
quantum registers (i.e., groups of qubits) whereas ρ(u l ) and

ρ ǫ k are two single-qubit states. The unitaries U 1 [(][k][)], U 0 [(][k][)] [†] act

on ρ [(] l− [k][)] 1 [, controlled by][ ρ][(][u] [l] [). The right-most operation (][SW][ ’s)]
swaps the states of ρ [(] l− [k][)] 1 [and][ σ] [k] [, controlled by][ ρ] [ǫ] k [.]


A. A subclass implementable on noisy gate-model
quantum devices


With a limited number of qubits and other current
quantum hardware restrictions, not all QR dynamics of
the form Eqs. (1)–(2) can be efficiently implemented.
Here we describe a subclass of the universal QR class
implementable on current gate-model quantum devices.


QRs in this subclass are governed by Eqs. (1)–(2) with

unitary evolutions T j [(][k][)] (ρ [(] l− [k][)] 1 [) =][ U] [ (] j [k][)] ρ [(] l− [k][)] 1 [U] [ (] j [k][)] [†] (j = 0, 1),

where the unitaries U 0 [(][k][)] and U 1 [(][k][)] are arbitrary but
fixed. In practice, U j [(][k][)] can be implemented by native quantum gates of the NISQ devices, possibly composed of single-qubit and multi-qubit gates each parameterized by some gate parameter. These gate parameters can be chosen arbitrarily but fixed and should
not generate trivial dynamics (e.g., we should not have
U 0 [(][k][)] = U 1 [(][k][)] ), thus precise tuning of these parameters
is not required. In Sec. VI A, we suggest some natural choices of U [(][k][)] tailored for the cloud-based IBM
j
quantum devices [31]. The QR dynamics in this subclass has a natural quantum circuit interpretation, see
Fig. 1. The state ρ(u l ) encodes the input u l as a classical
mixture ρ(u l ) = u l |0⟩⟨0| + (1 − u l )|1⟩⟨1|, meaning that

we apply U 0 [(][k][)] ρ l [(] − [k][)] 1 [U] [ (] 0 [k][)] [†] with probability u l, and apply



0 [(][k][)] [†] U 0 [(][k][)] U 1 [(][k][)] ρ [(] l− [k][)] 1 [U] [ (] 1 [k][)] [†]



0 [(][k][)] [†] U 0 [(][k][)] = U 1 [(][k][)] ρ [(] l− [k][)] 1 [U] [ (] 1 [k][)] [†]



U [(][k][)] [†]




[ (] 1 [k][)] [†] U 0 [(][k][)] [†]



1 with



4


B. Invariance under stationary Markovian
hardware noise and time-invariant readout error


The QR dynamics Eq. (1) is invariant under stationary
Markovian noise. A stationary Markovian noise process
acting on the k-th subsystem during some time interval
τ (l − 1) ≤ t ≤ τl, where l is the time step and τ > 0, can
be modeled as a CPTP map T [(][k][)] for all l ≥ 0. The k-th
subsystem’s dynamics Eq.(2) under this noise process is


ρ [(] l [k][)] = (1 − ǫ k ) u l T [(][k][)] ◦ T 0 [(][k][)] + (1 − u l )T [(][k][)] ◦ T 1 [(][k][)] ρ [(] l− [k][)] 1
� �

+ ǫ k T [(][k][)] (σ k ),


where T [(][k][)] ◦ T j [(][k][)] is again some CPTP for j = 0, 1 and
T [(][k][)] (σ k ) = σ k [′] [is again some fixed density operator. The]
resulting noisy dynamics again has the form Eq. (2) and
the form of QR dynamics Eq. (1) also remains unchanged.
That is, the universal family M is invariant and remains
universal under stationary Markovian noise. For hardware implementation of the QR subclass described in
Sec. IV A, if the hardware noise is stationary and Markovian, then it acts to replace U j [(][k][)] ρ [(] l− [k][)] 1 [U] [ (] j [k][)] [†] with another

(k)
CPTP map T j [(][ρ] [(] l− [k][)] 1 [). The resulting noisy QR dynam-]
ics is again of the form Eq. (1).
Stationary Markovian noise model is the noise model
adopted in the IBM Qiskit simulator [35, 36]. The Qiskit
noisy simulation approximates the hardware noise as a
CPTP map being applied after the application of a unitary gate. The noise parameters are estimated during
periodic calibrations on the hardware. Between two calibrations, the calibrated noise parameters remain unchanged and the noisy simulation approximates the hardware noise by a stationary Markovian noise model. However, during the experiments, the underlying hardware
noise could potentially be time-varying. Considering
these factors, the agreement between our experimental
and Qiskit noisy simulation results (see Appendix V D for
the data) indicate the underlying hardware noise approximately preserves the QR dynamics of the form Eq. (1)
during the experiments. If the underlying noise is nonstationary but changes slowly, the QR output weights
can be re-trained periodically using most recently gathered data. This remains challenging to be demonstrated
on current cloud-accessed only NISQ devices but can be
possible on future NISQ machines.

Furthermore, QR predicted outputs remain unchanged
under time-invariant readout error whenever a linear
readout function is used (i.e., R = 1 in Eq. (3), which is
often employed in practice and in our proof-of-principle
experiments). This is because time-invariant readout
error introduces a time-invariant linear transformation
of the measurement data and if the output weights
w ri 1i1,...,i,...,r n in and w c are optimized via linear regression, the
resulting QR predicted outputs y l remain unchanged; see
Appendix B for the derivation.



probability 1 − u l . Let ρ [(] l− [k][)] 1 [denote the QR’s][ k][-th sub-]
system state after these operations. The state ρ ǫ k is a
classical mixture ρ ǫ k = (1 − ǫ k )|0⟩⟨0| + ǫ k |1⟩⟨1| that encodes the rate ǫ k at which the k-th subsystem forgets
its initial conditions. That is, with probability ǫ k, the
states ρ [(] l− [k][)] 1 [and][ σ] [k] [ are exchanged, equivalent to resetting]

the state ρ [(] l− [k][)] 1 [to the fixed density operator][ σ] [k] [; otherwise]

the state ρ [(] l− [k][)] 1 [is unchanged with probability 1][ −] [ǫ] [k] [. We]
again associate the readout function Eq. (3) to this QR
subclass.


V. REALIZATION OF A SUBCLASS ON

CURRENT QUANTUM HARDWARE


We present two implementation schemes of the subclass described in Sec. IV A on current gate-model quantum computers, such as on the IBM superconducting
quantum devices. The first scheme takes into account
limitations of some current hardware, and the second
scheme employs quantum non-demolition (QND) measurements to substantially reduce the number of circuit
runs required. We further show that QR’s convergence
property leads to more efficient versions of both schemes.
Here, we focus on n-qubit QRs with a single subsystem
(N = 1 in Eq. (1)) and drop the subsystem index k in
Eq. (2). The case with multiple subsystems (N > 1) is
a straightforward extension. We may choose σ = |ψ⟩⟨ψ|
with an easy to prepare pure state |ψ⟩. In all schemes,
we initialize the QR circuits in |0⟩ [⊗][n] .
The first implementation follows from an earlier work

[37, Sec. III] and is employed in our proof-of-principle experiments (see Sec. VI). We consider NISQ devices that
allow pure state preparation. Instead of realizing Fig. 1
that requires mixed state preparation, we efficiently implement QRs through Monte Carlo sampling. We construct N m circuits, such that for each circuit and at each
timestep l, we apply U 0 and U 1 with probabilities (1−ǫ)u l
and (1 − ǫ)(1 − u l ), respectively; otherwise the circuit is
set in |ψ⟩ with probability ǫ. Therefore, for each N m circuits and each time l, implementing the input-dependent
QR dynamics T (u l ) in Eq. (1) amounts to applying the
gate sequence realizing U 0 or U 1, or resetting the circuit
in |ψ⟩. As N m is increased, the average of all measurements gives a more accurate estimate of the true expectation ⟨Z [(][i][)] ⟩ l . Furthermore, some current NISQ devices
do not allow qubit reset, meaning that once a qubit is
measured, it cannot be re-used for computation. To estimate ⟨Z [(][i][)] ⟩ l, we re-initialize N m circuits in |0⟩ [⊗][n] and
re-apply T (u k ) from time k = 1 to time k = l, and
only measuring Z [(][i][)] at the final time l. Each of the
N m circuits is run for S shots at each time l. To process a length-L input sequence under the pure state and
qubit re-set limitations requires N m SL circuit runs and
N m S(1+· · ·+L) = N m S(L+1)L/2 applications of T (u l ).
If qubit reset is available, a more efficient scheme using
QND measurements [38] can be realized, see Appendix C
for the details. We no longer need to re-run the N m circuits from time 1 to estimate ⟨Z [(][i][)] ⟩ l . Instead we just run
each of the N m circuits S shots, meaning that for each
circuit we perform a QND measurement of Z [(][i][)] at time l,
continue running the circuit until the next measurement,
and so forth. QND measurements ensure information
encoded in ρ l is retained from one timestep to the next.
This scheme requires N m SL applications of T (u l ) but
only N m S circuit runs as opposed to N m SL runs in the
first scheme. We remark that a recent noisy quantum
device is equipped with the qubit reset functionality [39],
and it will be interesting to implement this scheme in
such a device in a future work.



5


The QR’s convergence property (see Appendix A 1)
leads to more efficient versions of both schemes. Let
M ≥ 1 be a fixed integer and suppose that we want to estimate ⟨Z [(][i][)] ⟩ l at a sufficiently large time l (that depends
on ǫ, i.e., the rate of forgetting the initial condition).
Suppose we initialize N m circuits in |0⟩ [⊗][n], re-apply and
re-run T (u k ) from k = 1 as before. We then obtain the
QR states ρ l−M at time l − M and ρ l at time l. Thanks
to the convergence property, we can instead re-initialize
the N m circuits in |0⟩ [⊗][n] at time l −M and from this time
onwards re-apply and re-run T (u k ) according to inputs
{u l−M+1, . . ., u l }. At time l, we have the corresponding QR state ˜ρ l . By the convergence property (see Appendix C for the derivation), we can make the difference
between ρ l and ˜ρ l negligible by choosing M appropriately
based on ǫ. If we perform repeated measurements on ρ l
and ˜ρ l, the estimates of ⟨Z [(][i][)] ⟩ l and ⟨Z [�] [(][i][)] ⟩ l = Tr(˜ρ l Z [(][i][)] )
will also be close; see Appendix D.
The convergence property can be readily exploited on
current NISQ machines, leading to efficient versions of
both schemes. The first scheme now requires N m SL circuit runs but only N m SM applications of T (u l ). The
second scheme now only needs N m S circuit runs and
N m SM applications of T (u l ), both are independent of
the input length L, enabling QR’s potential for fast and
scalable temporal processing. In all schemes, it is possible and perhaps advantageous to set S = 1 and run
N m circuits (possibly in parallel if multiple copies of the
same hardware are available), for a sufficiently large N m .
The average of N m measurements estimates ⟨Z [(][i][)] ⟩, whose
estimation accuracy increases as N m increases; see Appendix D for the analysis. Since qubit reset is not yet
available on the IBM superconducting quantum devices,
we employ the first implementation scheme in our proofof-principle experiments. It will be a future work of interest to realize these more efficient protocols on gate-model
quantum hardware.


VI. PROOF-OF-PRINCIPLE EXPERIMENTS


Five nonlinear tasks are chosen to carefully test different computational aspects of the QR proposal. Tasks IIV have the fading memory property. Tasks I and II
test the QR’s ability to learn high-dimensional nonlinear
maps. Both tasks are governed by linear dynamics determined by some matrix A and have the same form of
nonlinear output. The maximum singular value σ max (A)
determines the rate at which the dynamics forgets its
initial condition while the sparsity of A reflects the pairwise correlation of the reservoir state elements. Task I
is described by a dense matrix A with σ max (A) = 0.5
and Task II is governed by A with 95% sparsity with
σ max (A) = 0.99. Task III tests the QR’s ability to learn
nonlinear maps governed by a highly nonlinear dynamics. Task IV tests the short-term memory ability and
Task V is a long-term memory map for testing the capability of the QR beyond its theoretical guarantee. For all


experimental and numerical details, see Appendix V.
We implement four distinct QRs from the subclass described in Sec. IV A on three IBM superconducting quantum processors [31]. Each QR consists of a single subsystem (N = 1 in Eq. (1)) with a linear output function (R = 1 in Eq. (3)). Hereafter, we drop the subsystem index k. A 4-qubit and a 10-qubit QRs are implemented on the 20-qubit Boeblingen device; qubits with
lower gate errors and longer coherence times are chosen.
The 5-qubit Ourense and Vigo devices are used for two
distinct 5-qubit QRs. These 5-qubit quantum devices admit simpler qubit couplings but lower gate errors than the
20-qubit Boeblingen device; see Appendix V E for hardware specifications. Through comparison among the four
QRs, we can investigate the impact of the size of QRs,
the complexity of quantum circuits implementing the QR
dynamics and the intrinsic hardware noise on the QRs’
approximation performance.


A. Quantum circuits for QRs


We require the QRs to forget initial conditions for approximating fading memory maps. Traditionally, initial
conditions are washed-out with a sufficiently long input
sequence until reaching a steady state. Here we bypass
the washout by choosing σ = (|0⟩⟨0|) [⊗][n] and U 0 so that
|0⟩ [⊗][n] is the steady state of Eq. (1) under u l = 1, meaning that we can initialize the QR circuits in |0⟩ [⊗][n] . Furthermore, U 0 and U 1 should be different and hardwareefficient but sufficiently complex to produce non-trivial
quantum dynamics. We choose a circuit schematics (also
see Fig. 2(a) and (b)),



6


FIG. 2. Quantum circuit schematics for (a) U 0 (θ) and (b)
U 1 (φ) employed in proof-of-principle experiments, described
by Eq. (4) in Sec. VI A. Here j t and j c are the target and
control qubits, respectively. The unitaries U 0 (θ), U 1 (φ) consist of N 0, N 1 layers of highlighted gate operations, with each
layer acting on a different qubit pair (j t, j c ).


FIG. 3. Qubit coupling maps of the IBM superconducting
quantum processors. (a) The 20-qubit Boeblingen device. (b)
Both the 5-qubit Ourense and Vigo devices.


For the 4-qubit and 10-qubit QRs on the Boeblingen
device, we choose the number of layers N 0 = N 1 = 5 in
Eq. (4). For the 5-qubit Ourense QR, we implement a
simpler form of Eq. (4), given by



U 0 (θ) =


U 1 (φ) =



N 0
�

j=1



n
� U 3 [(][i][)] [(][φ] 0 i [)]


i=1



U 3 [(][j] [t] [)] (θ j t )CX j c j t U 3 [(][j] [t] [)] (θ j t ) [†] [�],
�



U 0 =



4
� CX j c j t, U 1 (φ) =

j=1



5
� U 3 [(][i][)] [(][φ] i [)][.]


i=1



(4)



�



n
� U 3 [(][i][)] [(][φ] j i [)CX] [j] c [j] t
� i=1



N 1
�

j=1



,



To implement a different QR dynamics on the 5-qubit
Vigo device, we choose



where θ j t = (θ [0] j t [,][ θ] [1] j t [,][ θ] [2] j t [) and][ φ] j i [= (][φ] [0] j i [,][ φ] [1] j i [,][ φ] [2] j i [) are]
gate parameters, each independently and uniformly randomly sampled from [−2π, 2π]. Here U 3 [(][i][)] is an arbitrary
rotation on single qubit i [40] with inverse U 3 [(][j] [t] [)] (θ j t ) [†] =
U 3 [(][j] [t] [)] (−θ [0] j t [,][ −][θ] [2] j t [,][ −][θ] [1] j t [), and CX] [j] c [j] t [is the CNOT gate]
with control qubit j c and target qubit j t . These quantum gates are native to the aforementioned IBM superconducting quantum processors, meaning that no further
decomposition into simpler gates is required to implement these chosen gates [31]. The numbers of layers N 0
and N 1 are sufficiently large to couple all qubits linearly
while respecting the coherence limits of these devices.
Owing to the more flexible qubit couplings in the Boeblingen device, circuits implementing the 4-qubit and 10qubit QRs have more gate and random parameters than
the 5-qubit QRs’.



U 0 (θ) =


U 1 (φ) =



3
� �R y [(][j] [t] [)] (θ j t )CX j c j t R y [(][j] [t] [)] (θ j t ) [†] [�],

j=1



5
� R x [(][i][)] [(][φ] i [)][.]


i=1



Here R y [(][i][)] and R x [(][i][)] [are rotational][ Y][ and][ X][ gates on qubit]
i, respectively. Both gates are special instances of the
arbitrary single-qubit rotational gate U 3 [(][i][)] with one (free)
gate parameter while the other two being fixed constants.
For all QRs, natively coupled control and target qubits
for the CNOT gates are chosen, meaning that a CNOT
gate can be directly applied to the qubit pair without
additional gate operations. See Fig. 3 for the device qubit
coupling maps and Appendix V B for the QR quantum
circuit details.


B. Experimental implementation


In this section we report on experiments demonstrating
the first implementation scheme described in Sec. V. We
choose a sufficiently large N m = 1024 and ǫ = 0.1 for
a moderate short-term memory. To estimate ⟨Z [(][i][)] ⟩ l at
time l, each of the N m circuits implementing the QRs on
the Boeblingen device and the 5-qubit QRs are run for
S = 1024 and S = 8192 shots, respectively. These shot
numbers are chosen according to circuit execution times
of the devices.
We apply the four QRs to the five nonlinear tasks on
the multi-step ahead prediction and map emulation problems. To implement the same washout as for the QRs
for each target map, we inject a constant input sequence
u l = 1 of length 50 followed by train and test inputs uniformly randomly sampled from u l ∈ [0, 1]. This change
in the input statistics leads to a transitory target output
response. We remove the associated transients by discarding the first four target input-output data and the
corresponding QR experimental data, see Appendix V C
for all data. For the multi-step ahead problem, train and
test time steps run from l = 5 to L T = 23 and L T +1 = 24
to L = 30, respectively. For the map emulation problem,
K = 2 train input-output pairs running from l = 5 to
L [′] = 24 are used, followed by one unseen test inputoutput pair with the same time steps. The number of
train and test data in our proof-of-principle experiments
is limited by the length of quantum circuits allowed on
the IBM quantum processors. Furthermore, these cloudbased quantum processors are shared among users, making continuous experiments infeasible and durations of
experiments lengthy. Yet our work indicates that despite
these current limitations, NISQ devices can demonstrate
learning of input-output maps and supports QR as a viable intermediate application of NISQ machines on the
road to full-fledged quantum devices equipped with quantum error correction.
To harness the flexibility of the QR approach, a multitasking technique is used, in which the four QRs are
evolved and the estimates of ⟨Z [(][i][)] ⟩ l for all time steps
are recorded once, whereas the readout parameters w
are optimized independently for each task. That is a
fixed QR dynamics, with fixed gate parameter values, is
exploited for multiple tasks simultaneously. We evaluate
and compare the task performance of QRs using the normalized mean-squared error between prediction y| L T +1:L
and target y| L T +1:L, computed as



NMSE =



L
� |y l − y l | [2] /∆ [2] y [,]

l=L T +1



1 L
where µ = L−L T � l=L T +1 [y] [l] [, ∆] [2] y [=][ �] [L] l=L T +1 [(][y] [l] [ −] [µ][)] [2] [.]
While the success of experimental demonstration of hybrid quantum-classical algorithms often requires error
mitigation techniques to reduce the effect of decoherence

[41, 42], we remark that our results are obtained without
any process or readout error mitigation.



7


C. QR task performance


As the number of qubits increases, the 10-qubit Boeblingen QR is expected to perform better than other
QRs. For the multi-step ahead prediction problem, we
observe that two qubits in the 10-qubit Boeblingen QR
experienced significant time-varying deviations between
the experimental data and simulation results on the
Qiskit simulator; see Appendix V D for a discussion. To
remedy this issue, we set the corresponding elements
of w to be zeros. The resulting 10-qubit Boeblingen
QR (with NMSE<0.08) outperforms other QRs with a
smaller number of qubits on the first four tasks, and
achieves an almost two-fold performance improvement on
Tasks II and III; see Table I for all NMSEs on the multistep ahead prediction problem. The 10-qubit Boeblingen
QR predicted outputs follow the target outputs relatively
closely as shown in Fig. 4(a). The 5-qubit Ourense QR
admits very simple dynamics, whereas the 5-qubit Vigo
QR has more gate operations and gate parameters. The
5-qubit Ourense QR is outperformed by the 5-qubit Vigo
QR in all tasks. Considering that the Ourense and Vigo
devices have similar noise characteristics and the same
qubit coupling map, this suggests the QR performance
can be improved by choosing a more complex quantum
circuit, in the sense of having a longer gate sequence.


TABLE I. NMSEs on the multi-step ahead prediction.

Task 10-qubit 4-qubit 5-qubit 5-qubit
Boeblingen Boeblingen Ourense Vigo

I 0.051 0.088 0.24 0.070

II 0.072 0.12 0.68 0.22

III 0.043 0.10 0.25 0.081

IV 0.079 0.092 0.34 0.11

V 0.47 0.41 2.3 0.20


TABLE II. NMSEs on the map emulation.


Task Multiplexed 5-qubit 5-qubit

QR Ourense Vigo

I 0.20 0.26 0.32

II 0.13 0.27 0.23

III 0.16 0.46 0.26

IV 0.25 0.30 0.36

V 0.20 1.1 0.17


The 10-qubit Boeblingen QR performs better on all
tasks than the 5-qubit QRs except on Task V. This could
be due to the impact of the higher noise level in the Boeblingen device and the fact that the output sequence is
generated by a map that is not known to be fading memory, see Appendix V E for the hardware specifications.
Our universal class of QRs can exploit the property of
spatial multiplexing as initially proposed in Ref. [22]; also
see [24] and Fig. 5 for an illustration. Outputs of dis

8


FIG. 4. (a) Shows the QRs’ predicted outputs for the multi-step prediction problem, rows and columns correspond to different
tasks and QRs, respectively. (d) Shows the QRs’ predicted outputs for the map emulation problem, first column corresponds
to the multiplexed QR.


achieve comparable performance to the individual members as well as gaining an almost two-fold performance
boost on Tasks II and III. We anticipate that spatial
multiplexing of QRs with more complex circuit structures and a larger number of qubits can lead to further
performance improvements.


VII. CONCLUSION



FIG. 5. The spatial multiplexing schematic. The same input
sequence is injected into two distinct 5-qubit QRs. The internal states Tr(ρ l Z [(][i][)] ) of the two QRs are linearly combined to
form a single output.


tinct and non-interacting 5-qubit QRs can be combined
linearly to harness the computational features of both
members. Since the combined Ourense and Vigo devices
have 10 qubits overall as with the 10-qubit Boeblingen
QR but with lower noise levels, it would be meaningful
to combine the 5-qubit Vigo and Ourense QRs via spatial
multiplexing on the map emulation problem. The results
of this multiplexing is summarized in Table II.
The combination of two 5-qubit QRs as discussed
above achieves NMSE = 0.20, 0.13, 0.16, 0.25, 0.20 for the
five tasks without any readout or process error mitigation. The predicted multiplexed QR outputs corresponding to the unseen inputs follow the target outputs relatively closely as shown in Fig. 4(b). Without spatial
multiplexing, the 5-qubit Ourense or the 5-qubit Vigo
QR show a worse performance in the first four tasks; see
Table II. The spatial multiplexed 5-qubit QR combines
computational features from the constituent QRs and can



We propose a novel class of quantum reservoir computers endowed with universality property that is implementable on available noisy gate-model quantum hardware for temporal information processing. Our approach
can harness arbitrary but fixed quantum circuits native
to noisy quantum processors, without precise tuning of
the circuit parameters. Our theoretical analysis is supported by proof-of-concept experiments on current superconducting quantum devices, demonstrating that smallscale noisy quantum reservoirs can perform non-trivial
nonlinear temporal processing tasks under current hardware limitations, in the absence of readout and process
error mitigation techniques. We also detail more efficient
implementation schemes of our QR proposal that could
enable QR’s potential for fast and scalable temporal processing. It is a future work of interest to realize these
more efficient protocols on quantum hardware. Our work
indicates that quantum reservoir computing can serve as
a viable intermediate application of NISQ devices on the
road to full-fledged quantum computers.
Our approach is scalable in the number of qubits by
offloading exponentially costly computations to noisy
quantum systems and utilizing classical algorithms with


a linear (in the number of qubits) computational cost to
process sequential data. Moreover, when implemented on
NISQ devices, the micro-second timescale for the evolution of the quantum reservoir suggests its potential for
real-time fast signal processing tasks. Guided by our
theory, we applied the spatial multiplexing technique initially proposed in [22], and demonstrate experimentally
that exploiting distinct computational features of multiple small noisy quantum reservoirs can lead to a computational boost. As NISQ hardware becomes increasingly
accessible and the noise level is continually reduced, we
anticipate that the quantum reservoir approach will find
useful applications in a broad range of scientific disciplines that employ time series modeling and analysis. We
are also optimistic for useful applications to be possible
even with a noise level above the threshold for continuous

quantum error correction.


VIII. ACKNOWLEDGMENTS


The authors thank Keisuke Fujii for an insightful discussion. NY is supported by the MEXT Quantum Leap
Flagship Program Grant Number JPMXS0118067285.


Appendix A: Universality for approximating
nonlinear fading memory maps


We first define notation for the rest of this section. Let K([0, 1]) be the set of infinite sequences u =
{. . ., u −1, u 0, u 1, . . .} such that u l ∈ [0, 1] for all l ∈ Z.
Let K [+] ([0, 1]) and K [−] ([0, 1]) be subsets of K([0, 1]) for
which the indices are restricted to Z [+] = {1, 2, . . .} and
Z [−] = {. . ., −2, −1, 0}, respectively. For any complex
matrix A, ∥A∥ p = Tr(√A [†] A p ) 1/p is the Schatten p-norm

for some p ∈ [1, ∞). For any operator T, the induced
operator norm is ∥T ∥ p−p = sup A∈C n×n,∥A∥ p =1 ∥T (A)∥ p .
Let D(2 [n] ) denotes the set of 2 [n] × 2 [n] density operators.


Consider an input-output map M that maps an infinite input sequence u ∈ K([0, 1]) to a real infinite
output sequence y ∈ K(R). We say that M is wfading memory if there exists a decreasing sequence
w = {w 0, w 1, . . .} with lim l→∞ w l = 0, such that for
any u, v ∈ K [−] ([0, 1]), we have |M (u) 0 − M (v) 0 | → 0
whenever sup l∈Z − |w −l (u l − v l )| → 0. Here M (u) l = y l is
the output sequence at time l. We also require M to be
causal and time-invariant as in Ref. [24], meaning that
the output of M at time l only depends on the input up to
and including time l, and its outputs are invariant under
time-shifts. Now we are interested in approximating M
with a time-invariant fading memory map M produced
by a quantum reservoir computer.



9


1. The convergence property


Since M is fading memory, the map M must also forget
its initial condition ρ 0 . This is the convergence property

[32] or the echo-state property [30]. We give a precise
definition here.


Definition 1 (Convergence). An input-dependent CPTP
map T is convergent with respect to input u ∈ K([0, 1])
if there exists a sequence {δ l ; l ≥ 0} with δ l - 0 and
lim l→∞ δ l = 0 such that for all u ∈ K [+] ([0, 1]), for any
two density operators ρ j,l (j = 1, 2) satisfying ρ j,l =
T (u l )ρ j,l−1, it holds that ∥ρ 1,l − ρ 2,l ∥ 1 ≤ δ l . If a QR
dynamic T is convergent, we call the QR a convergent
system.


Lemma 1. The QR dynamics given by Eqs. (1) and (2)
is convergent with respect to inputs u ∈ K([0, 1]).


Proof. First we show each subsystem governed by Eq. (2)
is convergent. For any ρ, σ ∈D(2 [n] ), u l ∈ [0, 1] and
ǫ k ∈ (0, 1], we have


∥T [(][k][)] (u l )(ρ − σ)∥ 1



where the first inequality follows from [33, Theorem 9.2]
and the convex combination u l T 0 [(][k][)] +(1 − u l )T 1 [(][k][)] is again
a CPTP map. Now let ρ 1,0 and ρ 2,0 be two arbitrary
initial density operators, using the inequality Eq. (A1) L
times, we have


∥ρ 1,L − ρ 2,L ∥ 1


←−

= � Ll=1 [T] [ (][k][)] [(][u] [l] [)] (ρ 1,0 − ρ 2,0 )
����� � ���� 1

≤ (1 − ǫ k ) [L] ∥ρ 1,0 − ρ 2,0 ∥ 1 ≤ 2(1 − ǫ k ) [L],


L
where [←−] � l=1 [T] [ (][k][)] [(][u] [l] [) is the time-composition of][ T] [ (][k][)] [(][u] [l] [)]
from right to left.


Secondly, we show that the QR dynamics Eq. (1) is
convergent by showing that T (u l ) = [�] [N] k=1 [T] [ (][k][)] [(][u] [l] [) is]
again convergent when the subsystems are initialized
in a product state. We apply the same argument as
in [43, Lemma 5]. Consider two CPTP maps T [(1)] (u l )
and T [(2)] (u l ) of the form Eq. (2). Let ρ 1,0 ⊗ σ 1,0 and
ρ 2,0 ⊗ σ 2,0 be two arbitrary initial product states. Then
T [(1)] (u l ) ⊗ T [(2)] (u l ) is again convergent with respect to all



= (1 − ǫ k ) u l T 0 [(][k][)] + (1 − u l )T 1 [(][k][)] (ρ − σ)
���� � ��� 1
≤ (1 − ǫ k )∥ρ − σ∥ 1 ≤ 2(1 − ǫ k ),



(A1)


10



u ∈ K([0, 1]), as shown in the following,


∥ρ 1,L ⊗ σ 1,L − ρ 2,L ⊗ σ 2,L ∥ 1


←−
L

≤ � l=1 [T] [ (1)] [(][u] [l] [)][ ⊗] [T] [ (2)] [(][u] [l] [)] (ρ 1,0 ⊗ σ 1,0 − ρ 2,0 ⊗ σ 1,0 )
����� � ���� 1


←−
L

+ � l=1 [T] [ (1)] [(][u] [l] [)][ ⊗] [T] [ (2)] [(][u] [l] [)] (ρ 2,0 ⊗ σ 1,0 − ρ 2,0 ⊗ σ 2,0 )
����� � ���� 1


←−

= � Ll=1 [T] [ (1)] [(][u] [l] [)] (ρ 1,0 − ρ 2,0 ) ∥σ 1,L ∥ 1
����� � ���� 1


←−
L

+ � l=1 [T] [ (2)] [(][u] [l] [)] (σ 1,0 − σ 2,0 ) ∥ρ 2,L ∥ 1
����� � ���� 1

≤ 2(1 − ǫ 1 ) [L] + 2(1 − ǫ 2 ) [L] .


Repeating this argument N times shows the QR dynamics T (u l ) = [�] [N] k=1 [T] [ (][k][)] [(][u] [l] [) is again convergent.]


2. The fading memory property


Associate the readout function Eq. (3) to the QR dynamics Eqs. (1) and (2). This defines an input-output
(I/O) map M (T,h w ) .This I/O map is causal, meaning that
the its output y l only depends on u l ′ for l [′] ≤ l. Furthermore, it is time-invariant, meaning that TODO:DEFN
HERE.

y τ +l = M (T,h w ) (S τ (u)) l for all τ ∈ Z, where S τ (u) =
{. . ., u τ −1, u τ, u τ +1 . . .} shifts the input sequence by τ .
By causality and time-invariance, it suffices to consider
the outputs y l of M (T,h w ) (u) l for l ≤ 0 and left-infinite
inputs u ∈ K [−] ([0, 1]); see [24, 27, 29] for details.
For any u ∈ K [−] ([0, 1]) and any initial condition ρ −∞,




[0, 1] and A ∈ C [2] [nk] [×][2] [nk],


∥T [(][k][)] (x) − T [(][k][)] (y)∥ 1−1



where the last inequality follows from [44, Theorem 2.1].
We remark that [24, Lemma 3] is stated with respect to
the Schatten p = 2 norm, but the same argument holds
for Schatten p = 1 norm.


3. The universality property


Now consider the family M of maps M (T,h w ) . We state
our main universality result.


Theorem 2 (Universality). For any null sequence w, the
QR class M is dense in C(K [−] ([0, 1]), ∥· ∥ w ). That is,
given any w-fading memory map M ∈ C(K [−] ([0, 1]), ∥·
∥ w ) and any δ > 0, there exists M (T,h w ) ∈M such that
for all u ∈ K [−] ([0, 1]), sup l∈Z − |M (u) l −M (T,h w ) (u) l | < δ.


We apply the Stone-Weierstrass Theorem to show that
M is dense in C(K [−] ([0, 1]), ∥· ∥ w ) . It has been shown
that the space (K [−] ([0, 1]), ∥· ∥ w ) is a compact metric
space [27, Lemma 2]. We now state the Stone-Weierstrass
Theorem.


Theorem 3 (Stone-Weierstrass). Let E be a compact
metric space and C(E) be the set of real-valued continuous functions defined on E. If a subalgebra A of C(E)
contains the constant functions and separates points of
E, then A is dense in C(E).


Proof of Theorem 2. The family M forms a polynomial
algebra follows from [24, Lemma 5] and the observation that for any QR dynamics T 1 (u l ) = [�] [N] k=1 [1] [T] [ (] 1 [k][)] (u l )
and T 2 (u l ) = [�] [N] k=1 [2] [T] [ (] 2 [k][)] (u l ), where each T 1 [(][k][)], T 2 [(][k][)] has
the form Eq. (2), we again have T (u l )(ρ 1 ⊗ ρ 2 ) =
T 1 (u l )ρ 1 ⊗ T 2 (u l )ρ 2 is of the form Eq. (1). Furthermore,
T (u l ) = T 1 (u l ) ⊗ T 2 (u l ) is again convergent when initialized in a product state of the subsystems. Therefore,
the family M forms a polynomial algebra consisting of
w-fading memory maps.
Constant functions can be obtained by setting
w ri 1i1,...,i,...,r n in = 0 in Eq. (3). It remains to show that M
separates points in K [−] ([0, 1]). That is, for any distinct u, v ∈ K [−] ([0, 1]) with u l ̸= v l for at least one
l, we need to find a map M (T,h w ) ∈M such that
M (T,h w ) (u) 0 ̸= M (T,h w ) (v) 0 . We show that we can construct a single-qubit quantum reservoir with this property.



= sup
A∈C [2][nk][ ×][2][nk],∥A∥ 1 =1



T [(][k][)] (x) − T [(][k][)] (y) A
���� � ��� 1



= (1 − ǫ k )|x − y| sup
A∈C [2][nk][ ×][2][nk],∥A∥ 1 =1



k)
T (0 (A) − T 1 [(][k][)] (A)
��� ��� 1



k) k)
≤ (1 − ǫ k )|x − y| ����T (0 ��� 1−1 [+] ���T (1 ��� 1−1

≤ 2(1 − ǫ k )|x − y|,



�



M (T,h w ) (u) 0 = h w



−→

∞
� j=0 [T][ (][u] [−][j] [)] ρ −∞
�� �



,
�



∞
where [−→] � j=0 [T][ (][u] [−][j] [) = lim] [N] [→∞] [T][ (][u] [0] [)][ · · ·][ T][ (][u] [l][−][N] [) and]
the limit is point-wise. We can restate the fading memory property in terms of continuity of M (T,h w ) with
respect to a certain norm. Given a null sequence w
(i.e., a decreasing sequence w with lim l→∞ w l = 0) and
any u ∈ K [−] ([0, 1]), define a weighted norm ∥u∥ w =
sup l∈Z − |u l |w −l . The map M (T,h w ) is w-fading memory
if it is continuous in (K [−] ([0, 1]), ∥· ∥ w ).


Definition 2 (Fading memory). Given a null sequence
w, the set of w-fading memory maps is the set of
all continuous functions C(K [−] ([0, 1]), ∥· ∥ w ) defined on
(K [−] ([0, 1]), ∥· ∥ w ).


Lemma 2. For any null sequence w, M (T,h w ) induced by
QR described by Eqs. (1)–(3) is w-fading memory.


Proof. Using the same argument in [24, Lemma 3], it
follows that M (T,h w ) is w-fading memory if each k-th
subsystem dynamics T [(][k][)] (u l ) is continuous with respect
to the inputs u l ∈ [0, 1] for all k = 1, . . ., N . If fact, we
show that T [(][k][)] (u l ) is uniformly continuous. Let x, y ∈






11





 (A2)





T (u l ) = |00⟩⟨00| + (1 − ǫ)












0 0 0 0
sin [2] (2J)(2u l − 1) cos [2] (2J) 0 0
0 0 cos(2J) cos(2α) − cos(2J) sin(2α)
0 0 cos(2J) sin(2α) cos(2J) cos(2α)



Consider a single-qubit quantum reservoir with a linear
readout function (n = 1, R = 1, N = 1). For the rest of
this proof, we drop the subsystem index. This quantum
reservoir consists of one system qubit and one ancilla
qubit denoted as ρ a . Choose the dynamics



The above is a power series of the form



f (θ) = 2w 1 sin [2] (2J)(1 − ǫ)



∞
� θ [j] (u −j − v −j ),

j=0



ρ l = T (u l )ρ l−1
= (1 − ǫ) �u l Tr a �e [−][iH] (ρ l−1 ⊗ ρ [0] a [)][e] [iH] [�]

+(1 − u l )Tr a �e [−][iH] (ρ l−1 ⊗ ρ [1] a [)][e] [iH] [��] + ǫK I2 [,]



(A3)



where ρ [j] a [=][ |][j][⟩⟨][j][|][ for][ j][ = 0][,][ 1, Tr] [a] [denotes the par-]
tial trace over ancilla ρ a and ǫ ∈ (0, 1). The map
K I2 [is a CPTP map defined as][ K] [ I] 2 [(][X][) = Tr(][X][)] [ I] 2 [for]

any X ∈ C [2][×][2] . The Hamiltonian H is of the Ising
type H = J(X [(0)] X [(1)] + Y [(0)] Y [(1)] ) + α [�] [1] j=0 [Z] [(][j][)] [, where]
X [(][j][)], Y [(][j][)] and Z [(][j][)] are the Pauli X, Y and Z operators
on qubit j, with j = 0 being the ancilla qubit.
We order an orthogonal basis for C [2][×][2] as {I, Z, X, Y }.
The matrix representation of the CPTP map Eq. (A3)
is given by Eq. (A2). Since Eq. (A3) is convergent, we
can choose any initial condition ρ −∞ = |0⟩⟨0| with the
corresponding vector representation ρ ∞ = 2 [1] 1 1 0 0 .

� �

Taking a linear readout function, for u ∈ K [−] ([0, 1]), the
quantum reservoir implements



2 [(][X][) = Tr(][X][)] [ I] 2



2 [is a CPTP map defined as][ K] [ I] 2



2 [1] 1 1 0 0 .
� �



−→

∞
� j=0 [T] [(][u] [−][j] [)]
�� �



+ w c,

2



M (T,h w ) (u) 0 = 2w 1



ρ −∞



�



where [·] 2 is the second element of the vector corresponding to Tr(Zρ 0 )/2.
Now given two distinct inputs u, v ∈ K [−] ([0, 1]), suppose that u 0 ̸= v 0 . Then choose J such that cos [2] (2J) = 0
and therefore,


M (T,h w ) (u) 0 − M (T,h w ) (v) 0 = 2w 1 (1 − ǫ)(u 0 − v 0 ) ̸= 0.


Suppose u 0 = v 0, note that in general


M (T,h w ) (u) 0



= w 1 sin [2] (2J)(1 − ǫ)



∞
�

j=0



j
�(1 − ǫ) cos [2] (2J)� (2u −j − 1).



Choose ǫ ∈ (0, 1) and J such that (1 − ǫ) cos [2] (2J) ∈
(0, 1 − ǫ). Then the above is a convergent power series
and the subtraction is well-defined:


M (T,h w ) (u) 0 − M (T,h w ) (v) 0



where f (θ) has a nonzero radius of convergence and is
non-constant since θ = (1 − ǫ) cos [2] (2J) ∈ (0, 1 − ǫ), (1 −
ǫ) sin [2] (2J) ∈ (0, 1−ǫ) and u, v are assumed to be distinct.
Furthermore, since we assume that u 0 = v 0, we have
f (0) = 0. Invoking [45, Theorem 3.2], there exists β > 0
such that f (θ) ̸= 0 for all |θ| ≤ β, θ ̸= 0. This concludes
the proof for separation of points. The universality of M
now follows from the Stone-Weierstrass Theorem.


Appendix B: Invariance under time-invariant
readout error


The QR outputs are invariant under time-invariant
readout error whenever a linear readout function is used.
That is when R = 1 in Eq. (3), the QR predicted outputs
y l remain unchanged under time-invariant readout error.
Let B = {|i⟩} be the computational basis for an n-qubit
system, with i = 1, . . ., 2 [n] . The readout error is characterized by a measurement calibration matrix A whose
i, j-th element A i,j = Pr(i|j) is the probability of measuring the state |i⟩∈B given that the state is prepared
in the state |j⟩∈B.
We employ the readout error correction method described in Ref. [4]. For an n-qubit QR, at each time step
l, we execute 2 [n] calibration circuits with each circuit initialized in one of the 2 [n] computational basis elements.
The outcomes are used to create the measurement calibration matrix A l . The readout error at time step l
is corrected by applying the pseudo-inverse of A l to the
measured outcomes from the experiments.
For all experiments, the measurement outcomes are
stored as the count of measuring each basis elements
in B. Let v l = �v [1] l [· · ·][ v] [2] l [n] 1�, where v [i] l [is the]

count of measuring |i⟩ ∈ B at time step l. Let
z l = �⟨Z [(1)] ⟩ l - · · ⟨Z [(][n][)] ⟩ l 1�, where ⟨Z [(][i][)] ⟩ l is the finite
sampled approximation of ⟨Z [(][i][)] ⟩ l for i = 1, . . ., n. Then
we have z l = v l B, where B is a linear transformation. After applying readout error correction, we have
z [′] l [=][ v] [l] [A] [+] l [B][, where][ A] [+] l [is the pseudo-inverse of][ A] [l] [. To]
optimize the readout function parameters w, collect all

⊤
measurement data in a matrix v = �v [⊤] 1 - · · v [⊤] L � so

⊤
that z = �z [⊤] 1 - · · z [⊤] L � = vB, where L is the sequence



= 2w 1 sin [2] (2J)(1 − ǫ)



∞
�

j=0



j
�(1 − ǫ) cos [2] (2J)� (u −j − v −j ).


12



length. The linear output of the quantum reservoir computer is y = vBw, where w includes the bias term w c .
Append a corresponding row and column to A [†] l [to ac-]
count for the bias term. Suppose the readout error is
time-invariant, then A [+] = A [+] l for l = 1, . . ., L. The
quantum reservoir computer output after readout error
correction is y [′] = vA [+] Bw [′] . Assume that A [+] has all rows
linearly independent, then ordinary least squares yields
Bw [′] = ABw. Now given test data v test with readout
error correction applied, v test A [+] Bw [′] = v test A [+] ABw =
v test Bw. Therefore, the QR predicted outputs are invariant under time-invariant readout error.


Appendix C: Efficient implementations of a subclass
on gate-model quantum computers


We detail the second (more efficient) implementation
scheme described in Sec. V and show how the QR’s convergence property leads to more efficient versions of both
schemes in Sec. V.

If qubit reset is available, we can implement the second scheme in Sec. V based on quantum non-demolition
(QND) measurements [38]. In this scheme, to estimate
⟨Z [(][i][)] ⟩ l, we no longer need to re-initialize and re-run the
N m circuits from time 1. Instead, for each N m circuits we
perform a QND measurement of Z [(][i][)] at time l, continue
running the circuit until the next QND measurement,
and so forth. QND measurements ensure the information
encoded in ρ l is retained from one timestep to the next.
To process a length-L input sequence, each N m circuits
is run S shots so that the average of N m S measurements
at time l estimates ⟨Z [(][i][)] ⟩ l . That is, this scheme needs
N m SL applications of T (u l ), but only N m S circuit runs
compared to N m LS runs in the first scheme (see Sec. V).
This presents a substantial saving as the number of circuit runs is independent of the input sequence length L.
To explain QND measurements, we first show that direct measurement of Z on a “system” qubit is equivalent
to coupling the qubit with an ancilla qubit via CNOT and
measuring Z a, the Pauli Z operator acting on the ancilla
qubit “a” [33]. To see this, let |ψ⟩ sys = α|0⟩ sys + β|1⟩ sys
be the state of the system qubit. Prepare the ancilla
qubit at the ground state |0⟩ a . We write


CNOT = |0⟩⟨0| sys ⊗ I a + |1⟩⟨1| sys ⊗ X a,


where I a and X a are the identity and Pauli X operators
acting on the ancilla qubit. The system and ancilla state
after applying CNOT is


|Ψ⟩ = CNOT|ψ⟩ sys ⊗|0⟩ a = α|00⟩ + β|11⟩.


Measurement of Z a on the ancilla qubit is described by
the projectors P + = I sys ⊗|0⟩⟨0| a and P − = I sys ⊗|1⟩⟨1| a .



C C

|0⟩ [⊗][n] ✌✌ |0⟩ [⊗][n] ✌✌


FIG. 6. Quantum circuit implementing the QND measurements by coupling ancilla qubits |0⟩ [⊗][n] with the QR system
qubits |ψ⟩ l−1 .


Therefore, the probabilities and post-measurement sys
tem states are

Pr(+) = ⟨Ψ|P + |Ψ⟩ = |α| [2], [Tr] [a] [ (][P] Pr(+) [+] [|][Ψ][⟩⟨][Ψ][|][P] [+] [)] = |0⟩⟨0| sys,

Pr(−) = ⟨Ψ|P − |Ψ⟩ = |β| [2], [Tr] [a] [ (][P] [−] [|][Ψ] − [⟩⟨][Ψ][|][P] [−] [)] = |1⟩⟨1| sys,

Pr( )


where Tr a (·) is the partial trace over the ancilla qubit.
Now for an n-qubit QR, we associate each system qubit
in the QR with its ancilla qubit. All n ancilla qubits are
prepared in the ground state. Suppose that when restricted to pure state preparation, we have drawn N m
circuits using Monte Carlo sampling. For each of the
N m circuits and each time step l, we apply the aforementioned ancilla-coupled measurement of Z [(][i][)] for each
system qubit in the QR. After measuring the n ancilla
qubits, we reset and re-prepare them in the ground state
for measurements at next time l + 1; see Fig. 6.
In Fig. 6, |ψ⟩ l−1 denotes the state of the system (QR)
qubits and |0⟩ [⊗][n] denotes the ancilla qubits initialized
in the groud state. Here we have grouped the system
and ancilla qubits and represent them using single wires.
The unitary operator U l [′] [is][ U] [0] [ or][ U] [1] [ with probabilities]
(1 − ǫ)u l and (1 − ǫ)(1 − u l ) and U (l) = U l [′] [C][, where][ C][ is a]
product of n CNOT gates each acting on the i-th systemancilla qubit pair. Measuring Z a [(][i][)] on the i-th ancilla
qubit and resetting it at each time step l = 1, . . ., L is
equivalent to having L ancilla qubits associated to the ith system qubit and measuring Z a [(][i],l [)] [(i.e., the][ l][-th ancilla]
qubit associated to the i-th system qubit). The resulting
QR dynamics is


T (u l )ρ l−1 = (1 − ǫ) (u l T 0 + (1 − u l )T 1 ) ρ l−1 + ǫσ,


where T j (ρ l−1 ) = Tr A (U j Cρ l−1 ⊗ (|0⟩⟨0|) [⊗][n] C [†] U j [†] [) for]
j = 0, 1, and Tr A (·) is the partial trace over all n ancilla qubits denoted by “A”.
We now show that the measured observables Z a,l [(][i][)] [com-]
mute at different times as required by QND. More generally, we will show that Z a,l = [�] [n] i=1 [O] a [(][i],l [)] [(][l][ = 1][, . . ., L][),]

where for each i we have O a [(][i],l [)] [=][ I] [(][i][)] [ (the identity operator]

on the i-th qubit) or O a [(][i],l [)] [=][ Z] a [(][i],l [)] [, are QND observables.]
Firstly, we have the commutator [Z a,k, Z a,j ] = 0 for all
k, j = 1, . . ., L. Denote the evolved observables in the
Heisenberg picture by


Z a (l) = U (1) [†] - · · U (l) [†] Z a,l U (l) · · · U (1) = U l [†] :1 [Z] [a][,l] [U] [l][:1] [,]



|ψ⟩ l−1 U l [′]



U l [′] +1


where U l:1 = U (l) · · · U (1). For k, j = 1, . . ., L with j <
k, we have


[Z a (j), Z a (k)]

= U j [†] :1 [Z] [a][,j] [U] [j][:1] [U] [ †] k:1 [Z] [a][,k] [U] [k][:1] [ −] [U] [ †] k:1 [Z] [a][,k] [U] [k][:1] [U] [ †] j:1 [Z] [a][,j] [U] [j][:1]

= U j [†] :1 [Z] [a][,j] [U] [ †] k:j+1 [Z] [a][,k] [U] [k][:1] [ −] [U] [ †] k:1 [Z] [a][,k] [U] [k][:][j][+1] [Z] [a][,j] [U] [j][:1]

= U k [†] :1 [[][Z] [a][,j] [, Z] [a][,k] []][U] [k][:1] [ = 0][,]


where in the second last equality we have used the fact
that Z a,j commutes with the future unitary operations
U k:j+1 . If j > k, apply the same argument as above
shows [Z a (j), Z a (k)] = −[Z a (k), Z a (j)] = 0. The commutativity of Z a (j) and Z a (k) for all j, k ≥ 1 means that
the sequence {Z a (j), j = 1, 2, . . .} has a joint probability
distribution and constitutes a classical stochastic process.
QND measurements on the sequence gives a realization
of this stochastic process.
The QR’s convergence property (see Appendix A 1)
leads to more efficient versions of both schemes in Sec. V.
Let M, ρ l, ρ l−M and ˜ρ l be as given in Sec. V. By the convergence property (Eq. A1), we have


˜
∥ρ l − ρ l ∥ 1 ≤ (1 − ǫ) [M] ∥ρ l−M − (|0⟩⟨0|) [⊗][n] ∥ 1 ≤ 2(1 − ǫ) [M] .
(C1)
The difference between ρ l and ˜ρ l can be made negligible by choosing M appropriately based on ǫ. If
we perform repeated measurements on ρ l and ˜ρ l, then
the finite-sample estimates of ⟨Z [�] [(][i][)] ⟩ l = Tr(˜ρ l Z [(][i][)] ) and
⟨Z [(][i][)] ⟩ l = Tr(ρ l Z [(][i][)] ) will also be close; see Appendix D.
Using the convergence property, the first scheme in Sec. V
requires N m SL circuit runs but only N m SLM applications of T (u l ). When L > M, a substantial saving in
the number of applications of T can be obtained (for
timesteps l > M ) compared to the previous quadratic
dependence on L. The second scheme now only requires
at most N m SM applications of T (u l ) and N m S circuit
runs, both are independent of the input sequence length
L. This provides a path for fast and large scale temporal
processing using QRs.


Appendix D: Monte Carlo estimation


For all schemes described in Sec. V, we can set S = 1
and run N m Monte Carlo sampled circuits (possibly in
parallel if many copies of the same hardware are available) for a sufficiently large N m . We show that the average of all N m measurements at time l estimates ⟨Z [(][i][)] ⟩ l
and its variance vanishes as N m tends to infinity.
First consider estimating ⟨Z [(][i][)] ⟩ l by re-initializing each
N m circuit in |0⟩ [⊗][n] and re-running them from time 1 to
time l according to inputs {u 1, . . ., u l }. Recall that


⟨Z [(][i][)] ⟩ l = Tr(Z [(][i][)] ρ l ) = Tr(Z [(][i][)] T (u l ) · · · T (u 1 )(|0⟩⟨0|) [⊗][n] ),


where T (u k ) is the input-dependent CPTP map defined
in Eq. (1) for k = 1, . . ., l. Define independent discrete


13


valued random variables X k such that


Pr(X k = 0) = (1 − ǫ)u k,
Pr(X k = 1) = (1 − ǫ)(1 − u k ),
Pr(X k = 2) = ǫ.


To implement the QR, for each time k, we independently
sample N m random variables X k,j (j = 1, . . ., N m ) from
the same distribution as X k . Define



T x =








T 0, if x = 0,

T 1, if x = 1,

K σ, if x = 2,



where K σ (ρ) = σ is a constant CPTP map that sends
any density operator ρ to the constant density operator
σ in Eq. (2). The random CPTP maps T X k,j follow the
same distribution as X k,j and are independent for each
k and j. Furthermore, E[T X k,j ] = T (u k ).
For the j-th circuit, we implement a sequence of (random) CPTP maps T X l,j - · · T X 1,j so that at time l, the
(random) QR state is

ρ [X] [l,j] = T X l,j      - · · T X 1,j (|0⟩⟨0|) [⊗][n],


where X l,j = (X 1,j, . . ., X l,j ). For each j-th circuit, we
measure Z [(][i][)] and denote its random outcome by Z [(][i][)] l,j .
Note that for j = 1, . . ., N m, Z [(][i][)] l,j are independent (but
not necessarily identically distributed) random variables
taking values ±1 (eigenvalues of Z [(][i][)] ) with conditional
probabilities (conditional on the random variables X l,j )



Pr
�



Z [(][i][)] l,j = z|X l,j = Tr ρ [X] [l,j] P z [(][i][)], z = ±1,
� � �



where P ± [(][i] 1 [)] [are the projectors such that][ Z] [(][i][)] [ =][ P] +1 [ (][i][)] [−][P] − [ (][i] 1 [)] [.]
Consider the average of all N m measurement outcomes,
by the law of total expectation,



1

N m



N m
� E �

j=1



Z [(][i][)] l,j
�



1

=
N m


1

=
N m


1

=
N m


1

=
N m



N m
� E �E �

j=1



N m
� E �Tr �Z [(][i][)] ρ [X] [l,j] ��

j=1


N m
� Tr �Z [(][i][)] E[T X l,j ] · · · E[T X 1,j ](|0⟩⟨0|) [⊗][n] [�]

j=1


N m
� Tr �Z [(][i][)] T (u l ) · · · T (u 1 )(|0⟩⟨0|) [⊗][n] [�]

j=1



Z [(][i][)] l,j |X l,j
��



= Tr(ρ l Z [(][i][)] ) = ⟨Z [(][i][)] ⟩ l,


therefore the finite-sample estimate is unbiased. Moreover, using the fact that



E
��



2 [�]
Z [(][i][)] l,j � = � z [2] Pr �

z=±1



2 [�]
Z [(][i][)] l,j � = �



Z [(][i][)] l,j = z = 1,
�


the variance of the average of N m measurements is



Z [(][i][)] l,j 



Var







 N [1]



N m



N m
�

j=1



1
Z [(][i][)] l,j =
� N m



1

=
N m [2]



N m
�



� Var �

j=1



�1 −⟨Z [(][i][)] ⟩ l [2] � .



Using the convergence property, to estimate ⟨Z [(][i][)] ⟩ l for
a sufficiently large l (that depends on ǫ), we re-initialize
N m circuits at time l − M and run the circuits accord
˜
ing to inputs {u l−M+1, . . ., u l }. Let ⟨Z [�] [(][i][)] ⟩ l = Tr(Z [(][i][)] ρ l )
where


˜
ρ l = T (u l ) · · · T (u l−M+1 )(|0⟩⟨0|) [⊗][n] .


In this setting, for the j-th circuit, we implement a sequence of (random) CPTP maps T X l,j - · · T X l−M +1,j so
that the (random) QR state at time l is


ρ X� l,j = T X l,j     - · · T X l−M +1,j (|0⟩⟨0|) ⊗n,


where X [�] l,j = (X l−M+1,j, . . ., X l,j ). Let Z [�] [(][i][)] l,j be the
random outcome of measuring Z [(][i][)] . The conditional
probabilities are


Pr Z [(][i][)] = z|X [�] l,j = Tr ρ X� l,j P z (i), z = ±1.
�� � � �


A similar argument as above shows that the average of
all N m measurements satisfies



14


V. EXPERIMENTAL AND NUMERICAL

DETAILS


A. Nonlinear temporal processing tasks


We give detailed descriptions of the five nonlinear temporal processing tasks. Tasks I and II are governed by
linear reservoirs with polynomial readout [27, 29], described by

x l = Ax l−1 + cu l
�y l = h(x l ),


where A ∈ R [2000][×][2000] and c ∈ R [2000] . To have shortterm or fading memory, we rescale the maximum singular
value σ max (A) = 0.5 for Task I and σ max (A) = 0.99 for
Task II, meaning that Task II retains the initial condition
and past inputs for a longer time duration. The sparsity
of A determines the pairwise correlation between reservoir state elements. We set A to be a full (dense) matrix
for Task I and 95% sparse for Task II. The readout function h is a degree 2 polynomial in the state elements.
Task III is a recently proposed classical reservoir computing model that achieves good performance in chaotic
system modeling [27], described by

x l = p(u l )x l−1 + q(u l )
�y l = w [T] x l,


where p(u l ) = [�] [4] j=0 [A] [j] [u] [j] l [and][ q][(][u] [l] [) =][ �] [2] j=0 [B] [j] [u] [j] l
are matrix-valued polynomials in the input u l, A j ∈
R [700][×][700] [ �] R [700][×][700] and B j ∈ R [700][×][1] [ �] R [700][×][1] . For
Task III, We rescale σ max (A j ) < 3 [1] [for all][ j][ so that both]

it exhibits short-term memory. Task IV is a Volterra
series with kernel order 5 and memory 2, commonly applied to model responses of nonlinear systems in control
engineering [29],



 = ⟨Z [�] [(][i][)] ⟩ l,





1

 =

N m




�1 −⟨Z [�] [(][i][)] ⟩ l [2] � .



E


Var







 N [1]



 N [1]



N m



N m







y l = w c +



N m
�

j=1


N m
�

j=1



�
Z [(][i][)] l,j


�
Z [(][i][)] l,j



2
� w [j] i [1] [,...,j] [i]

j 1,...,j i =0



i
� u l−j k .


k=1



5
�


i=1



The convergence property and Eq. C1 ensure that the
bias (in mean) vanishes exponentially fast,



E
������



������







 N [1]



N m
�

j=1



 −⟨Z [(][i][)] ⟩ l





N m



�
Z [(][i][)] l,j



= Tr Z [(][i][)] (˜ρ l − ρ l )
��� � ����


˜
≤∥ρ l − ρ l ∥ 1 ≤ 2(1 − ǫ) [M],


where we have used the fact that for any Hermitian matrix A, |Tr(Z [(][i][)] A)| ≤ σ max (Z [(][i][)] )∥A∥ 1, with σ max (Z [(][i][)] ) =
1 denotes the maximum singular value of Z [(][i][)] . This
shows that the bias can be exponentially suppressed by
choosing M appropriately based on ǫ, so that the estimates of ⟨Z [�] [(][i][)] ⟩ l and ⟨Z [(][i][)] ⟩ l are also close.



For the first three tasks, elements of A, A j, B and w are
uniformly randomly sampled from [−1, 1]. The constant
c and coefficients of readout function h are also sampled
from the same distribution. The same applies to the
kernel coefficients w i [j] [1] [,...,j] [i] and w c in Task IV.
Task V is a missile moving with a constant velocity in
the horizontal plane, a continuous-time long-term memory nonlinear map [46] described by

x˙ 1 = x 2 − 0.1 cos(x 1 )(5x 1 − 4x [3] 1 [+][ x] [5] 1 [)][ −] [0][.][5 cos(][x] [1] [)][u]
�x˙ 2 = −65x 1 + 50x [3] 1 [−] [15][x] 1 [5] [−] [x] [2] [−] [100][u,]


with y = x 2 . This missile dynamics is simulated using the (4, 5) Runge-Kutta formula in MATLAB, with a
sampling time of τ = 1/80 for 1 second.


B. Quantum circuits for QRs


We detail the circuits implementing the QR dynamics in our proof-of-principle experiments presented in
Sec. VI. The quantum circuits for the 4-qubit and 10qubit Boeblingen QRs are shown in Fig. 7. The quantum circuits for the 5-qubit Ourense and 5-qubit Vigo
QRs are shown in Fig. 8.


C. Full input-output sequential data


Since we bypass the washout for QRs by initializing
them in the state |0⟩ [n], this is equivalent to washing out
their initial conditions with a length L w constant input
sequence u l = 1. The same washout has been applied
to all nonlinear tasks. We have checked that L w = 50
is enough for all tasks to reach steady states given the
same initialization x 0 = 0. Particular caution has been
taken to washout Task IV, in which we set u l = 1 for
l = −2, −1. For each target map, we discard the first
four input-output sequence data points, and the corresponding QR experimental data, to remove the transitory
output response due to the change in input statistics. In
Fig. 9, we show the full washout, train and test inputoutput target sequences for both the multi-step ahead
prediction and the map emulation problems. Fig. 10
plots the full target output sequences, the train and test
QR outputs on the multi-step ahead prediction problem.
Fig. 11 plots the full target output sequences, the train
and test QR outputs on the map emulation problem. In
all figures, the transitory responses are indicated by dotted lines.


D. Measurement and simulation data


We simulate the four QRs using the IBM Qiskit simulator under ideal and noisy conditions. The noise models
used are obtained from the device calibration data. We
fetched the updated device calibration data each time a
job was executed on the hardware. The circuits simulated are the same as the circuits employed for the experiments and so is the number of shots. For the multistep ahead prediction problem, the 10-qubit Boeblingen
QR experienced a significant deviation from simulated results on qubits Q = 1, 8 (see Fig. 12), resulting in larger
NMSE = 0.26, 0.29, 0.068, 0.15, 6.1 for the four tasks. After setting the readout parameters w 1 = w 8 = 0 for
Q = 1, 8, this issue was circumvented at the cost of
using a fewer number of computational features. The
resulting 10-qubit Boeblingen QR still achieves performance improvement over other QRs with a smaller number of qubits on the multi-step ahead prediction problem
in the first three tasks. A time-invariant readout error in
qubit i linearly transforms the expectation ⟨Z [(][i][)] ⟩ l . The
QR predicted outputs are invariant under time-invariant
readout errors when using linear regression to optimize



15


w, w c as derived in Appendix B. However, for the 10qubit Boeblingen QR, the deviations in qubits Q = 1, 8
were time-varying. On the other hand, the 5-qubit Vigo
device experienced almost time-invariant deviations in
qubit Q = 0 as shown in Figs. 12 and 13, but this does
not affect the performance of this QR noticeably. The
experimental results of the 5-qubit Ourense QR follow
the noisy simulation results closely. For the map emulation problem, the experimental results of both 5-qubit
QRs follow the simulated results closely, with an almost
time-invariant shift in Q = 0 for the 5-qubit Vigo QR.


E. Hardware specifications


The experiments were conducted on the IBM 20qubit Boeblingen (version 1.0.0), 5-qubit Ourense (version 1.0.0) and 5-qubit Vigo (version 1.0.0) superconducting quantum processors [31]. The gate duration for an
arbitrary single-qubit rotation gate U 3 [40] is τ U 3 ≈ 71.1
ns for all qubits whereas the CNOT gate durations differ
for different qubits.
See Fig. 7 for the 4-qubit and 10-qubit Boeblingen QR
quantum circuits. The circuits are chosen such that both
QRs have the same number of layers in U 0 and U 1 . In
this setting, the maximum duration of a circuit executed
on the Boeblingen device is the same for both QRs. As
stated in the main article, the chosen qubits for the 4qubit QR and the 10-qubit QR on the Boeblingen device are Q = 0, 1, 2, 3 and Q = 0, 1, 2, 3, 5, 6, 7, 8, 10, 12.
These qubits were chosen due to their longer coherence
times, shorter CNOT gate durations, smaller gate and
readout errors. During the experiment, the maximum
readout error was 10 [−][2] and the maximum U 3 gate error implemented was 10 [−][3] . The maximum CNOT gate
error implemented was 4.3 × 10 [−][2] and the maximum
CNOT gate duration was τ CNOT ≈ 427 ns. We assume
that commuting gates can be executed in parallel. We
choose N 0 = N 1 = 5 numbers of layers for U 0 and U 1
in the 4-qubit and 10-qubit Boeblingen QRs. The maximum length of any input sequence (including the transient) for the multi-step ahead prediction and the map
emulation problems is L = 30. Therefore, the maximum numbers of U 3 gate executions and CNOT gate
executions is 5L = 5 × 30 = 150. The maximum duration of a circuit executed on the Boeblingen device was
150×(τ U 3 +τ CNOT ) ≈ 150×(71.1+427) = 74.7 µs, within
the coherence times (T 1, T 2 ) for most qubits chosen.
Fig. 8 shows the quantum circuits for the 5-qubit
Ourense and 5-qubit Vigo QRs. Owing to the more restricted qubit couplings in these 5-qubit devices, the circuits for the 5-qubit QRs are simpler than that of the
4-qubit and 10-qubit Boeblingen QRs. To combine different computational features for the spatial multiplexing
technique, we choose circuits that are sufficiently different for these two 5-qubit QRs. In particular, the 5-qubit
Vigo QR consists of single-qubit rotational Y gates in U 0
and single-qubit rotational X gates in U 1 . On the other


16





The 5-qubit Ourense device achieves the same order
of magnitude in readout errors, coherence times and
CNOT gate durations as the 20-qubit Boeblingen device,
but lower CNOT gate errors. For the Ourense device,
the maximum U 3 gate error and readout error implemented were 0.9×10 [−][3] and 4.1×10 [−][2], and the maximum
CNOT gate error implemented was 8 × 10 [−][3], a lower error compared to the Boeblingen device. The maximum
CNOT gate duration implemented was τ CNOT ≈ 576
ns. For the 5-qubit Ourense QR, the circuit implementing U 0 is longer than that for U 1 . The U 0 circuit consists of four CNOT gates, and the maximum duration
of a circuit executed for the 5-qubit Ourense QR was
4L × τ CNOT ≈ 70 µs, also within the coherence limits of
most qubits.


The 5-qubit Vigo device is similar to the 5-qubit
Ourense device. They have the same qubit couplings
and share similar noise profile and hardware specifications. Rotational X and Y gates were used on this device, with gate duration τ = 35.5 ns. The maximum
single-qubit gate error implemented was 0.8 × 10 [−][3] and



tively. For this QR, U 0 is the longer circuit consisting
of three layers of single-qubit rotation Y gates and two
layers of CNOT gates. Therefore, the maximum duration of a circuit implemented was (3τ + 2τ CNOT )L =
(3 × 35.5 + 2 × 462.2) × 30 ≈ 31 µs.


17


FIG. 9. Full washout, train and test input-output sequences for (a) The multi-step ahead prediction problem and (b) The map
emulation problem. The first row in (a) and (b) shows the input sequences.


FIG. 10. The full target output sequences, the train and test output sequences of the four QRs for each task on the multi-step
ahead prediction problem. Each column corresponds to each n-qubit QR outputs. Each row corresponds to each task.


18


FIG. 11. The full target output sequences, the train and test output sequences of the QRs for each task on the map emulation
problem. (a) Shows the two train output sequences. (b) Shows the test output sequence. The columns correspond to the
multiplexed 5-qubit QRs, 5-qubit Ourense QR and the 5-qubit Vigo QR from the left to the right. Each row corresponds to
each task.


19


FIG. 12. Input sequence, experimental and simulation results for each qubit of the four QRs at each time step l = 1, . . ., 30,
for the multi-step ahead prediction problem.


20


FIG. 13. Experimental and simulation results for each qubit i = 0, . . ., 4 and each time step l = 1, . . ., 24, for the map emulation
problem. Three input sequences are used in this problem, labeled as inputs I, II and III. Row i in a sub-figure corresponds
to the experimental data for the i-th input sequence. Column j corresponds to the experimental data for the j-th qubit. (a)
Shows the experimental data for the 5-qubit Ourense QR. (b) Shows the experimental data for the 5-qubit Vigo QR.


[1] J. Biamonte, P. Wittek, N. Pancotti, P. Rebentrost,
N. Wiebe, and S. Lloyd, Quantum machine learning, Nature 549, 195 (2017).

[2] C. Ciliberto, M. Herbster, A. D. Ialongo, M. Pontil,
A. Rocchetto, S. Severini, and L. Wossnig, Quantum machine learning: a classical perspective, Proceedings of the
Royal Society A: Mathematical, Physical and Engineering Sciences 474, 20170551 (2018).

[3] J. Preskill, Quantum Computing in the NISQ era and
beyond, Quantum 2, 79 (2018).

[4] V. Havl´ıˇcek, A. D. C´orcoles, K. Temme, A. W. Harrow,
A. Kandala, J. M. Chow, and J. M. Gambetta, Supervised learning with quantum-enhanced feature spaces,
Nature 567, 209 (2019).

[5] A. Kandala, A. Mezzacapo, K. Temme, M. Takita,
M. Brink, J. M. Chow, and J. M. Gambetta, Hardwareefficient variational quantum eigensolver for small
molecules and quantum magnets, Nature 549, 242
(2017).

[6] J. Miller and M. Hardt, Stable recurrent models, in Proceedings of the 2019 International Conference on Learning Representations (ICLR) (2019).

[7] J. Hanson and M. Raginsky, Universal approximation of
input-output maps by temporal convolution nets, in Proceedings of the 33rd annual conference on Neural Information Processing Systems (NeurIPS) (2019).

[8] M. Rigotti et al., The importance of mixed selectivity in
complex cognitive tasks, Nature 497, 585 (2013).

[9] R. Pascanu, T. Mikolov, and Y. Bengio, On the difficulty
of training recurrent neural networks, in International
conference on machine learning (2013) pp. 1310–1318.

[10] C. Du, F. Cai, M. A. Zidan, W. Ma, S. H. Lee, and
W. D. Lu, Reservoir computing using dynamic memristors for temporal information processing, Nature Communications 8, 2204 (2017).

[11] K. Vandoorne, P. Mechet, T. Van Vaerenbergh, M. Fiers,
G. Morthier, D. Verstraeten, B. Schrauwen, J. Dambre,
and P. Bienstman, Experimental demonstration of reservoir computing on a silicon photonics chip, Nature Communications 5, 3541 (2014).

[12] Q. Vinckier, F. Duport, A. Smerieri, K. Vandoorne,
P. Bienstman, M. Haelterman, and S. Massar, Highperformance photonic reservoir computer based on a coherently driven passive cavity, Optica 2, 438 (2015).

[13] J. Torrejon, M. Riou, F. A. Araujo, S. Tsunegi,
G. Khalsa, D. Querlioz, P. Bortolotti, V. Cros,
K. Yakushiji, A. Fukushima, et al., Neuromorphic computing with nanoscale spintronic oscillators, Nature 547,
428 (2017).

[14] L. Larger, A. Bayl´on-Fuentes, R. Martinenghi, V. S.
Udaltsov, Y. K. Chembo, and M. Jacquot, High-speed
photonic reservoir computing using a time-delay-based
architecture: Million words per second classification,
Physical Review X 7, 011015 (2017).

[15] L. Gonon, L. Grigoryeva, and J.-P. Ortega, Approximation bounds for random neural networks and reservoir
systems, arXiv preprint arXiv:2002.05933 (2020).

[16] J. Dambre, D. Verstraeten, B. Schrauwen, and S. Massar,
Information processing capacity of dynamical systems,
Scientific reports 2, 1 (2012).

[17] L. Gonon, L. Grigoryeva, and J.-P. Ortega, Memory and



21


forecasting capacities of nonlinear recurrent networks,
arXiv preprint arXiv:2004.11234 (2020).

[18] G. Tanaka, T. Yamane, J. B. H´eroux, R. Nakane,
N. Kanazawa, S. Takeda, H. Numata, D. Nakano, and
A. Hirose, Recent advances in physical reservoir computing: A review, Neural Networks (2019).

[19] K. Nakajima, Physical reservoir computingan introductory perspective, Japanese Journal of Applied Physics
59, 060501 (2020).

[20] W. Maass, T. Natschl¨ager, and H. Markram, Real-time
computing without stable states: A new framework for
neural computation based on perturbations, Neural Computation 14, 2531 (2002).

[21] K. Fujii and K. Nakajima, Harnessing disorderedensemble quantum dynamics for machine learning, Phys.
Rev. Appl. 8, 024030 (2017).

[22] K. Nakajima, K. Fujii, M. Negoro, K. Mitarai, and
M. Kitagawa, Boosting computational power through
spatial multiplexing in quantum reservoir computing,
Physical Review Applied 11, 034021 (2019).

[23] M. Negoro, K. Mitarai, K. Fujii, K. Nakajima, and
M. Kitagawa, Machine learning with controllable quantum dynamics of a nuclear spin ensemble in a solid, arXiv
preprint arXiv:1806.10910 (2018).

[24] J. Chen and H. I. Nurdin, Learning nonlinear input–
output maps with dissipative quantum systems, Quantum Information Processing 18, 198 (2019).

[25] G. Cybenko, Approximation by superpositions of a sigmoidal function, Mathematics of control, signals and systems 2, 303 (1989).

[26] K. Hornik, M. Stinchcombe, and H. White, Multilayer
feedforward networks are universal approximators, Neural Networks 2, 359 (1989).

[27] L. Grigoryeva and J.-P. Ortega, Universal discrete-time
reservoir computers with stochastic inputs and linear
readouts using non-homogeneous state-affine systems,
The Journal of Machine Learning Research 19, 892
(2018).

[28] L. Grigoryeva and J. Ortega, Echo state networks are
universal, Neural Networks 108, 495 (2018).

[29] S. Boyd and L. Chua, Fading memory and the problem of
approximating nonlinear operators with Volterra series,
IEEE Trans. Circuits Syst. 32, 1150 (1985).

[30] H. Jaeger and H. Haas, Harnessing nonlinearity: Predicting chaotic systems and saving energy in wireless communications, Science 304, 5667 (2004).

[31] IBM Quantum Experience,
[https://www.ibm.com/quantum-computing/.](https://www.ibm.com/quantum-computing/)

[32] A. Pavlov, N. van de Wouw, and H. Nijmeijer, Convergent systems: Analysis and synthesis, in Control and Observer Design for Nonlinear Finite and Infinite Dimensional Systems, Lecture Notes in Control and Information
Science, Vol. 322, edited by T. Meurer, K. Graichen, and
E. D. Gilles (Springer, 2005) pp. 131–146.

[33] M. A. Nielsen and I. L. Chuang, Quantum Computation
and Quantum Information (New York: Cambridge University Press, 2010).

[34] J. Dieudonn´e, Foundations of Modern Analysis (Read
Books Ltd, 2013).

[35] Qiskit Aer: Device backend noise model simulations,
[https://github.com/Qiskit/qiskit-iqx-tutorials/blob/master/](https://github.com/Qiskit/qiskit-iqx-tutorials/blob/master/qiskit/advanced/aer/2_device_noise_simulation.ipynb)


[36] Qiskit Aer API Reference,
[https://qiskit.org/documentation/apidoc/aer.html.](https://qiskit.org/documentation/apidoc/aer.html)

[37] J. Chen, H. I. Nurdin, and N. Yamamoto, Single-input
single-output nonlinear system identification and signal
processing on near-term quantum computers, in Proceedings of the 2019 IEEE Conference on Decision and Control (CDC) (2019) pp. 401–406.

[38] V. B. Braginsky and F. Y. Khalili, Quantum measurement (Cambridge University Press, 1995).

[39] J. Pino, J. Dreiling, C. Figgatt, J. Gaebler, S. Moses,
C. Baldwin, M. Foss-Feig, D. Hayes, K. Mayer, C. RyanAnderson, et al., Demonstration of the QCCD trappedion quantum computer architecture, arXiv preprint
arXiv:2003.01293 (2020).

[40] A. W. Cross, L. S. Bishop, J. A. Smolin, and J. M. Gambetta, Open quantum assembly language, arXiv preprint
arXiv:1707.03429 (2017).

[41] A. Kandala, K. Temme, A. D. C´orcoles, A. Mezzacapo,



22


J. M. Chow, and J. M. Gambetta, Error mitigation extends the computational reach of a noisy quantum processor, Nature 567, 491 (2019).

[42] Y. Li and S. C. Benjamin, Efficient variational quantum
simulator incorporating active error minimization, Physical Review X 7, 021050 (2017).

[43] J. Chen and H. I. Nurdin, Correction to: Learning nonlinear input–output maps with dissipative quantum systems, Quantum Information Processing 18, 354 (2019).

[44] D. Perez-Garcia, M. M. Wolf, D. Petz, and M. B.
Ruskai, Contractivity of positive and trace-preserving
maps under l p norms, Journal of Mathematical Physics
47, 083506 (2006).

[[45] S. Lang, Complex Analysis, Graduate Texts in Mathe-](https://books.google.com.au/books?id=7S7vAAAAMAAJ)
matics (Springer-Verlag, 1985).

[46] X. Ni, M. Verhaegen, A. J. Krijgsman, and H. B. Verbruggen, A new method for identification and control of
nonlinear dynamic systems, Engineering Applications of
Artificial Intelligence 9, 231 (1996).


