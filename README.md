**Introduction**

In modern neurosurgery, intraoperative ECoG is the main method of neurophysiological monitoring during surgical treatment of pharmacoresistant epilepsy and epileptic syndrome complicating the course of other brain diseases: neoplasms, dysplasia, etc. Intraoperative ECoG makes it possible to determine the localization and extent of the epileptogenic zone in the cortex to be surgically removed (Kuruvilla et al., 2003; Tatum et al., 2008).

Hans Berger was the first person to record an electrocorticogram in humans in the early 1920s (Penfield et al., 1954; Kuruvilla et al., 2003). During surgical treatment of patients with cranial bone defects he recorded "brain waves" by placing electrodes on the dura mater. The foundations of intraoperative ECoG were laid by the works of neurosurgeon W.G. Penfield and neurophysiologist H.H. Jasper, performed in the 1940s. The method of neurosurgical treatment of epilepsy developed by Penfield consisted in destroying those parts of the cerebral cortex that represented the focus of seizure activity. Jasper substantiated and developed approaches to neurophysiological control: intraoperative recording of spontaneous cortical electrical activity and evoked activity during electrical stimulation of different brain regions (Penfield et al., 1954). Since the mid-1950s, ECoG monitoring has become the "gold standard" in surgical treatment of epilepsy. Current world practice shows that this method is used by 80-85% of neurosurgical clinics (Quesney et al., 2005; Tatum et al., 2008).

However, despite the sixty-year history of wide use of ECoG, many questions remain unanswered about the diagnostic and prognostic capabilities of this method, and sometimes its significance for neurosurgery is questioned. Thus, up to 15%-20% of neurosurgical clinics do not use ECoG (Quesney et al., 2005; Tatum et al., 2008).

**Literature review**

The current level of technological development has greatly contributed to the emergence of a new research direction in neuroscience, the experimental models of which involve the interpretation of measurements of brain activity in real time to form feedback signals or control commands for external devices (Kohler et al., 2017; Kramer et al., 2019). One of the most urgent tasks of this direction is development of techniques for building brain-computer interfaces (BCIs), the systems that provide direct control of external devices based on arbitrary modulation of brain activity. In fact, BCIs implement an additional channel of information exchange with the external environment, different from the natural pathway, which involves muscles and peripheral nerves (Abdulkader et al., 2015); thus, BCIs use intact brain capabilities while replacing the peripheral parts of the system that realizes movement.

In the same way as for the natural functioning of living organisms, feedback is necessary for the realization of movement with the help of the BCI. To create feedback, BCIs use artificially created stimuli that contain information about the current state of the system and make it possible to regulate it by means of arbitrary modulation of brain activity. Thus, a full-fledged information exchange channel should be bidirectional and allow not only transmitting commands from the brain to controlled external devices (e.g., prosthetic limbs), but also closing the feedback loop, providing the brain with real-time information about the current state of these devices (Lebedev, Ossadtchi, 2018).

Primarily, BCI systems are in demand in the tasks of restoring motor activity and communication among people whose motor functions have suffered as a result of injury or disease (Chaudhary et al., 2016). Electroencephalography (EEG) is the most widely used noninvasive technique for measuring brain activity in such systems (Machado et al., 2010). However, due to the fundamental limitations associated with indirect recording of neuronal activity, the bandwidth of the information channel realized with such EEG BCIs is usually rather low and does not exceed one bit per second (Mak et al., 2009; Waldert et al., 2016). Therefore, in most cases such BCIs can decode only a small number of discrete states (such as, for example, left arm movement, or resting state). Effective use of BCIs, especially in clinical practice, requires stability, accuracy, and - ultimately - the ability to decode continuous trajectories rather than discrete commands (Mak et al., 2009; Schalk, 2010), which requires at least a tenfold increase in the bandwidth of this communication channel.

Experience in the development of this problem shows that in the early stages of EEG-BCI systems development, new algorithmic solutions (Congedo et al., 2015) made it possible to achieve significant improvements in BCIs performance. Recently, however, the use of modern algorithmic approaches, in particular methods based on deep learning, brings only minor improvements in the performance of BCIs (Roy et al., 2019). Perhaps this indicates the existence of a limit to which we have approached in the task of decoding neuronal activity, which is especially evident in the case of non-invasive BCIs. Probably, further development of BCI algorithms is advisable in the direction of building interpretable architectures (Gilpin et al., 2018), capable of forming decisive rules amenable to analysis and interpretation (in contrast to the traditional model of using neural networks, in which the decision-making principle rather remains a "black box" and it is difficult to establish which features of initial data were determinative). In addition, to improve the quality of decoding, despite the limited amount of information that can be extracted from the electroencephalogram signal, it is necessary to develop methods that not only rely on learning procedures based on a specific set of measurements, but also allow for the consideration of a priori knowledge in the process of tuning the parameters of decoding algorithms (Gülçehre et al., 2016; Dagaev et al., 2017). Information about the physiological substrate of the arbitrary modulations of neuronal activity used in a particular BCI, the features of experimental models (Jayaram et al., 2016; Padfield et al., 2019), and the properties of the organism can be used as such a priori knowledge (Dagaev et al., 2017). Thus, on the one hand, it will be guaranteed compliance of the rules used with physiological principles, which is important, for example, when using BCIs in neurorehabilitation tasks. On the other hand, it will be possible to use the underlying architecture neural network architectures to extract new knowledge and discover hidden patterns in experimental data (Alain, Bengio, 2014).

The most radical and effective method of increasing BCI throughput is the use of invasive methods of recording brain activity. The data obtained as a result of such methods contain more complete information on movement parameters and already allow, for example, to implement control of complex prostheses with a large of freedom (Yanagisawa et al., 2012; Collinger et al., 2013). In particular, increased degrees of freedom in device control have been facilitated using invasive interfaces based on implantation of microelectrode arrays into the cortex (Kim et al., 2018), as evidenced by a number of studies (Collinger et al., 2013; Miranda et al., 2015). However, the use of such interfaces carries risks, associated with implantation (Kohler et al., 2017), and is limited to individual patients for whom specialized systems have been developed within the within the clinic (Miranda et al., 2015), as well as animal studies (Carmena et al., 2003; Velliste et al., 2008). A developing area now is the use of electrocorticogram (ECoG) technology, in which electrodes are placed on the surface of the brain subdurally or epidurally (under or on top of the dura mater without compromising the integrity of the cortex (Schalk & Leuthardt, 2011).

Electrocorticography is a significantly safer method compared to implantation of microelectrode arrays and is universally used in clinical practice for localization of epileptic focus, determination of tumor boundaries and mapping of functionally irreducible cortex (Hill et al., 2012). Meanwhile, ECoG is a promising method for realization of IMC due to higher signal stability during long-term use in comparison with intracortically implanted electrodes (Shokoueinejad et al., 2019), low noise level and high spatial resolution while covering relatively large cortical area (Kellis et al., 2016), availability of measurement of high frequency activity, which reflects local interaction of neurons in cortex (Schalk & Leuthardt, 2011). Other factors include the possibility of attracting many patients monitored using electrocorticography, who do not need to be exposed to additional risks of implantation for research.

ECoG signal characteristics such as high spatial resolution, low noise, few oculographic and myographic artifacts, and source proximity make it possible to detect with high accuracy the beginning of the motor act, distinguish the movements of individual fingers, decode the speed and direction of movement, and use the interface to control a complex prosthetic hand (Ball et al. 2009; Kubanek et al. 2009; Yanagisawa et al. 2011; Chestek et al. 2013; Hotson et al. 2016). However, none of the above studies implemented continuous motion decoding in real time.

Use of BCI, as a rule, assumes adjustment of parameters of the solver or the decoder under the individual user that allows to guarantee the maximum achievable accuracy of such devices. However, as it is known (Mühl et al., 2014), improvement of interface performance requires not only adaptation of parameters of decoding algorithms, but also training of the human, the interface user, in the model of operant conditioning by feedback signal based on the correct or incorrect actions performed by the control object. Having a system that simultaneously adapts the decoding algorithm and the user should significantly increase the efficiency of such tuning and provide high interface performance with short learning times (Zander et al., 2011).

The most convincing demonstration of the performance of neural interfaces is the use of such systems in real time. For this purpose, in the work of to create invasive interfaces in clinical settings involving limited patient interaction time, it requires a concentration of methodological and software tools combined with patient selection techniques and experimental user learning models.

The process of developing BCIs, especially those based on invasive technologies such as ECoG, involves close interaction between the development scientists and the clinical partners. Patients who are medically implanted with electrodes to localize epileptogenic zones or mark functionally irreducible cortical areas participate in research about creating invasive neurointerfaces. Such collaboration opens opportunities for the creation and validation of new clinical procedures that minimize patient risks and improve the quality of medical services. For example, passive intraoperative nonresponsive cortical mapping systems (Schalk et al., 2008; Korostenskaja et al., 2015) are being actively developed to replace procedures using direct electrical cortical stimulation, which often lead to seizures, critical patient condition and change of surgical plan. The introduction of safe techniques into the practice of clinical centers, as well as the development of new signal processing algorithms and protocols for the presentation of relevant functional stimuli to increase the accuracy of nonresponsive cortical mapping and improve the ergonomics of this procedure represent another relevant and socially relevant area of research, closely intertwined with the topic of neurocomputer interfaces (Sinkin et al., 2019).

Thus, in the last decades a great number of works appeared, which implement decoding of parameters of motion from ECoG, with different localization of electrodes, experimental models, and algorithms of signal processing. These studies have shown the possibility of detecting movements of the hand (Pistohl et al. 2012; Bleichner et al. 2016), fingers (Kubanek et al. 2009; Hotson et al. 2016), tongue and lips (Graimann et al. 2003; Miller et al. 2007), and feet (Satow et al. 2003).

**Methodology**

Our goal is to create a convolutional neural network decoder that has a similar performance to HTNet(Peterson et al. 2021). We received a dataset from National Institute of Mental Health, Neurology and Neurosurgery – (henceforth OKITI): the study population was 16 patients that were implanted with ECoG to diagnose drug-resistant epilepsy. We used 64 channels for the model training at the minimum, but some subjects have nearly twice that number . However, one patient S09 had some unused channels denoted by x. The electrode configuration consisted of strips and grids (figure 3). We can read them as 1D and 2D arrays. To record the dataset patients were asked to perform specific movements: tapping with their finger, foot, hand, tongue. One patient tapped with both right and left fingers, feet, and hands. Sampling rate was either 1024Hz or 2048Hz. We used two datatypes of u16 and s32. The length of the movements can be determined with timestamps that is also recorded in excel file.

![A close-up of a human skull

Description automatically generated with low confidence](Aspose.Words.23a8b37e-8801-4110-a5ca-c253a4f56cac.001.jpeg)

**Figure 1: patient 0: strips and grid**

Our programming language of choice was Python and we set up a Conda virtual environment with all the necessary libraries. The main libraries were TensorFlow and Keras since it would be best to compare with HTNet which also uses the same libraries. Then in the code we read the dataset and preprocess the data to delete channels contaminated with artifacts and EEG channels. We also filter 50 Hz harmonics and restrict data to 0-200 Hz range. After standardizing the data, we create a 2D convolutional neural network that has four convolutional layers and Dense layers (figure 2). After the training on GPU, we evaluate the best model by comparing predicted and test results. To do that we make a confusion matrix and find out the real-world accuracy.

![](Aspose.Words.23a8b37e-8801-4110-a5ca-c253a4f56cac.002.png)![Diagram

Description automatically generated](Aspose.Words.23a8b37e-8801-4110-a5ca-c253a4f56cac.003.png)

**Figure 2: OKITI model**

We also tried to replicate a study on HTNet which is based on densely connected neural networks that can be applied for transfer learning. We set up a Google Colab environment for that. HTNet is innovative because it will find the overlapping areas of electrodes for the best prediction, and it uses Hilbert transform. The architecture consists of three convolutional layers: 1D temporal convolution as a band-pass filtering, a depthwise convolution as a spatial filter, and a separable convolution as a classifier of temporal features. The author makes use of the Hilbert transform layer after the first convolutional layer to compute spectral power at data-driven frequencies. There are also pooling, dense, and other layers (figure 3).

![](Aspose.Words.23a8b37e-8801-4110-a5ca-c253a4f56cac.004.png)![Diagram

Description automatically generated](Aspose.Words.23a8b37e-8801-4110-a5ca-c253a4f56cac.005.png)

**Figure 3: neural network model of the first patient in HTNet**

**Results and Discussion**

We received great confusion matrices (figures 4-5) and an accuracy of roughly 90% (figure 6). In particular, the model predicted 100% accurately for four subjects which can be ascribed to overfitting. Nonetheless, this high accuracy is possible because of our small dataset, so the accuracy must be lower on a larger dataset. In addition, there was an error with a third subject which shows that there is a problem with encoding of the data and reading it correctly. We solved it by individual indexing to overcome an error.

![Text

Description automatically generated](Aspose.Words.23a8b37e-8801-4110-a5ca-c253a4f56cac.006.png)

**Figure 4: Subject 00 confusion matrix and accuracy**

![Text

Description automatically generated](Aspose.Words.23a8b37e-8801-4110-a5ca-c253a4f56cac.007.png)

**Figure 5: Subject 01 confusion matrix and accuracy**

![](Aspose.Words.23a8b37e-8801-4110-a5ca-c253a4f56cac.008.png)

**Figure 6**

The replication of HTNet was not ideal since it takes days to train the model and there is a limit in Google Colab where GPU can be used only for 12 hours continuously. We decided to limit the number of epochs to 2 since the author of HTNet also made graphs with just 2 epochs. It took 3 hours to train, but we were able to only reproduce one graph since we have not used the model for the unseen patients yet (figure 7).

![Graphical user interface, calendar

Description automatically generated with medium confidence](Aspose.Words.23a8b37e-8801-4110-a5ca-c253a4f56cac.009.png)![Application

Description automatically generated with medium confidence](Aspose.Words.23a8b37e-8801-4110-a5ca-c253a4f56cac.010.png)

**Figure 7: replicated plot (original on the right)**

We experimented on decoding synchronously recorded ECoG signals in offline mode. The results of this study allowed us to measure the quality of decoding achievable by the developed algorithms and showed the advantage of using deep learning to solve this problem.

The main limitation of the study is a quantity of subjects since there are few patients implanted with ECoG. It is likely that the development of long-term implantation technology over time will increase the amount of data and provide new opportunities for research in this area, but experience with both invasive and noninvasive interfaces suggests that customizing the system for a particular user and working with it will remain individualized.

**Conclusion**

The performance of invasive brain-computer interfaces can be improved using deep learning algorithms since it allows to automatically identify features and model patterns in data at different levels of complexity, and provide additional capabilities, such as extracting information from unlabeled data or applying transfer learning technology.

In the future we will make decoding of ECoG signals in real time and fine-tune HTNet parameters for OKITI dataset for comparison. We will try train with balanced class weights and solve an overfitting problem.

**References**

Abdulkader, S. N., Atia, A., & Mostafa, M. S. M. (2015). Brain computer interfacing: Applications and challenges. Egyptian Informatics Journal, 16(2), 213-230.

Alain, G., & Bengio, Y. (2014). What regularized auto-encoders learn from the data-generating distribution. The Journal of Machine Learning Research, 15(1), 3563-3593.

Ball, T., Schulze-Bonhage, A., Aertsen, A., & Mehring, C. (2009). Differential representation of arm movement direction in relation to cortical anatomy and function. Journal of neural engineering, 6(1), 016006.

Bleichner, M. G., Freudenburg, Z. V., Jansma, J. M., Aarnoutse, E. J., Vansteensel, M. J., & Ramsey, N. F. (2016). Give me a sign: decoding four complex hand gestures based on high-density ECoG. Brain Structure and Function, 221(1), 203-216.

Carmena, J. M., Lebedev, M. A., Crist, R. E., O'Doherty, J. E., Santucci, D. M., Dimitrov, D. F., ... & Nicolelis, M. A. (2003). Learning to control a brain–machine interface for reaching and grasping by primates. PLoS biology, 1(2), e42.

Chaudhary, U., Birbaumer, N., & Ramos-Murguialday, A. (2016). Brain–computer interfaces for communication and rehabilitation. Nature Reviews Neurology, 12(9), 513.

Chestek, C. A., Gilja, V., Blabe, C. H., Foster, B. L., Shenoy, K. V., Parvizi, J., & Henderson, J. M. (2013). Hand posture classification using electrocorticography signals in the gamma band over human sensorimotor brain areas. Journal of neural engineering, 10(2), 026002.

Collinger, J. L., Wodlinger, B., Downey, J. E., Wang, W., Tyler-Kabara, E. C., Weber, D. J., ... & Schwartz, A. B. (2013). High-performance neuroprosthetic control by an individual with tetraplegia. The Lancet, 381(9866), 557-564.

Congedo, M., & Barachant, A. (2015, January). A special form of SPD covariance matrix for interpretation and visualization of data manipulated with Riemannian geometry. In AIP Conference Proceedings (Vol. 1641, No. 1, pp. 495-503). American Institute of Physics.

Dagaev N., Volkova K. and Ossadtchi A. Latent variable method for automatic adaptation to background states in motor imagery BCI. J Neural Eng. 2017.

Gilpin, L. H., Bau, D., Yuan, B. Z., Bajwa, A., Specter, M., & Kagal, L. (2018, October). Explaining explanations: An overview of interpretability of machine learning. In 2018 IEEE 5th International Conference on data science and advanced analytics (DSAA) (pp. 80-89). IEEE.

Graimann, B., Huggins, J. E., Schlogl, A., Levine, S. P., & Pfurtscheller, G. (2003). Detection of movement-related patterns in ongoing single-channel electrocorticogram. IEEE Transactions on neural systems and rehabilitation engineering, 11(3), 276-281.

Gülçehre, Ç., & Bengio, Y. (2016). Knowledge matters: Importance of prior information for optimization. The Journal of Machine Learning Research, 17(1), 226-257.

Hotson, G., McMullen, D. P., Fifer, M. S., Johannes, M. S., Katyal, K. D., Para, M. P., ... & Crone, N. E. (2016). Individual finger control of a modular prosthetic limb using high-density electrocorticography in a human subject. Journal of neural engineering, 13(2), 026017.

Jayaram, V., Alamgir, M., Altun, Y., Scholkopf, B., & Grosse-Wentrup, M. (2016). Transfer learning in brain-computer interfaces. IEEE Computational Intelligence Magazine, 11(1), 20-31.

Kim, G., Kim, K., Lee, E., An, T., Choi, W., Lim, G., & Shin, J. (2018). Recent progress on microelectrodes in neural interfaces. Materials, 11(10), 1995.

Kohler, F., Gkogkidis, C. A., Bentler, C., Wang, X., Gierthmuehlen, M., Fischer, J., ... & Ball, T. (2017). Closed-loop interaction with the cerebral cortex: a review of wireless implant technology. Brain-Computer Interfaces, 4(3), 146-154.

Korostenskaja, M., Kamada, K., Guger, C., Salinas, C. M., Westerveld, M., Castillo, E. M., ... & Elsayed, M. (2015). Electrocorticography-based real-time functional mapping for pediatric epilepsy surgery. Journal of Pediatric Epilepsy, 4(04), 184-206.

Kramer, D. R., Kellis, S., Barbaro, M., Salas, M. A., Nune, G., Liu, C. Y., ... & Lee, B. (2019). Technical considerations for generating somatosensation via cortical stimulation in a closed-loop sensory/motor brain-computer interface system in humans. Journal of Clinical Neuroscience, 63, 116-121.

Kubanek, J. O. J. W. G. S. J., Miller, K. J., Ojemann, J. G., Wolpaw, J. R., & Schalk, G. (2009). Decoding flexion of individual fingers using electrocorticographic signals in humans. Journal of neural engineering, 6(6), 066001.

Kuruvilla A., Flink R. (2003). Intraoperative electrocorticography in epilepsy surgery: useful or not?  Seizure. Vol. 12. 577–584.

Lebedev, M. A., & Ossadtchi, A. (2018). Bidirectional neural interfaces. In Brain–Computer Interfaces Handbook (pp. 701-720). CRC Press.

Machado, S., Araújo, F., Paes, F., Velasques, B., Cunha, M., Budde, H., ... & Piedade, R. (2010). EEG-based brain-computer interfaces: an overview of basic concepts and clinical applications in neurorehabilitation. Reviews in the Neurosciences, 21(6), 451-468.

Mak, J. N., & Wolpaw, J. R. (2009). Clinical applications of brain-computer interfaces: current state and future prospects. IEEE reviews in biomedical engineering, 2, 187-199.

Miller, K. J., Leuthardt, E. C., Schalk, G., Rao, R. P., Anderson, N. R., Moran, D. W., ... & Ojemann, J. G. (2007). Spectral changes in cortical surface potentials during motor movement. Journal of Neuroscience, 27(9), 2424-2432.

Miranda, R. A., Casebeer, W. D., Hein, A. M., Judy, J. W., Krotkov, E. P., Laabs, T. L., ... & Weber, D. J. (2015). DARPA-funded efforts in the development of novel brain–computer interface technologies. Journal of neuroscience methods, 244, 52-67.

Mühl, C., Allison, B., Nijholt, A., & Chanel, G. (2014). A survey of affective brain computer interfaces: principles, state-of-the-art, and challenges. Brain-Computer Interfaces, 1(2), 66-84.

Padfield, N., Zabalza, J., Zhao, H., Masero, V., & Ren, J. (2019). EEG-based brain-computer interfaces using motor-imagery: Techniques and challenges. Sensors, 19(6), 1423.

Penfield, W., & Jasper, H. (1954). *Epilepsy and the functional anatomy of the human brain.* Little, Brown & Co..

Peterson, S. M., Steine-Hanson, Z., Davis, N., Rao, R. P. N., & Brunton, B. W. (2021).

Generalized neural decoders for transfer learning across participants and recording modalities.

Journal of Neural Engineering. https://doi.org/10.1088/1741-2552/abda0b

Pistohl, T., Schulze-Bonhage, A., Aertsen, A., Mehring, C., & Ball, T. (2012). Decoding natural grasp types from human ECoG. Neuroimage, 59(1), 248-260.

Quesney, L.F., Niedermeyer E. (2005). Electroencephalography. Basis, principles, clinical applications related fields. Philadelphia-Baltimore-NY: Lippincott Williams & Wilkins. 769–776.

Roy, Y., Banville, H., Albuquerque, I., Gramfort, A., Falk, T. H., & Faubert, J. (2019). Deep learning-based electroencephalography analysis: a systematic review. Journal of neural engineering, 16(5), 051001.

Satow, T., Matsuhashi, M., Ikeda, A., Yamamoto, J., Takayama, M., Begum, T., ... & Hashimoto, N. (2003). Distinct cortical areas for motor preparation and execution in human identified by Bereitschaftspotential recording and ECoG-EMG coherence analysis. Clinical neurophysiology, 114(7), 1259-1264.

Schalk, G. (2010). Can electrocorticography (ECoG) support robust and powerful brain-computer interfaces?. Frontiers in neuroengineering, 3, 9.

Schalk, G., & Leuthardt, E. C. (2011). Brain-computer interfaces using electrocorticographic signals. IEEE reviews in biomedical engineering, 4, 140-154.

Schalk, G., Miller, K. J., Anderson, N. R., Wilson, J. A., Smyth, M. D., Ojemann, J. G., ... & Leuthardt, E. C. (2008). Two-dimensional movement control using electrocorticographic signals in humans. Journal of neural engineering, 5(1), 75.

Sinkin, M. V., Osadchii, A. E., Lebedev, M. A., Volkova, K. V., Kondratova, M. S., Trifonov, I. S., & Krylov, V. V. (2019). Passive speech mapping of high accuracy during operations for gliomas of the dominant hemisphere. Neurosurgery, 21(3), 37-43.

Tatum W.O., Vale F.L., Anthony K.U. (2008).  A practical approach to neurophysiologic intraoperative monitoring. NY: Demos. 283–301.

Velliste, M., Perel, S., Spalding, M. C., Whitford, A. S., & Schwartz, A. B. (2008). Cortical control of a prosthetic arm for self-feeding. Nature, 453(7198), 1098.

Waldert, S. (2016). Invasive vs. non-invasive neuronal signals for brain-machine interfaces: will one prevail?. Frontiers in neuroscience, 10, 295.

Yanagisawa, T., Hirata, M., Saitoh, Y., Goto, T., Kishima, H., Fukuma, R., ... & Yoshimine, T. (2011). Real-time control of a prosthetic hand using human electrocorticography signals. Journal of neurosurgery, 114(6), 1715-1722.

Yanagisawa, T., Hirata, M., Saitoh, Y., Kishima, H., Matsushita, K., Goto, T., ... & Yoshimine, T. (2012). Electrocorticographic control of a prosthetic arm in paralyzed patients. Annals of neurology, 71(3), 353-361.

Zander, T. O., & Kothe, C. (2011). Towards passive brain–computer interfaces: applying brain–computer interface technology to human–machine systems in general. Journal of neural engineering, 8(2), 025005.
2

