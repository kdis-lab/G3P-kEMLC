# G3P-kEMLC: A Grammar-Guided Genetic Programming algorithm to create Ensembles of Multi-Label Classifiers

G3P-kEMLC is Grammar-Guided Genetic Programming (G3P) algorithm designed to build Ensembles of Multi-Label Classifiers (EMLCs).
Given the use of G3P, a tree-shaped ensemble is obtained, where at each node of the tree the prediction of children nodes are combined, and the leaves are the base multi-label classifiers.
Each of the base classifiers is focused on a subset of the labels of size _k_ (a.k.a. _k_-labelset). Unlike in other EMLCs, G3P-kEMLC is able to deal with base classifiers focused on _k_-labelsets of different size.
More information about this algorithm will be provided soon.

In this repository we provide the code of G3P-kEMLC, distributed under the GPLv3 License. G3P-kEMLC has been implemented using JCLEC [[Ven08]](#Ven08), Mulan [[Tso11]](#Tso11), and Weka [[Hal09]](#Hal09)  libraries. Besides, the latest release [(v 2.0)](https://github.com/kdis-lab/G3P-kEMLC/releases/tag/v2.0) provides the executable jar to execute G3P-kEMLC.

To execute G3P-kEMLC, the following command have to be executed:
```sh
java -jar G3P-kEMLC.jar configFile.cfg
```

The configuration file is a xml file including the parameters of the G3P algorithm. In this case, there are some parameters that are mandatory, which are presented in the following example.

```xml
<experiment>
  <process algorithm-type="g3pkemlc.Alg">
    <rand-gen-factory seed="10"/>
     
    <population-size>50</population-size>
    <max-of-generations>200</max-of-generations>    
     
    <recombinator rec-prob="0.5" />
    <mutator mut-prob="0.5" />
     
    <min-k>3</min-k>
    <max-k>7</max-k>
    <max-depth>3</max-depth>
    <max-children>7</max-children>
    <v>10</v>
    <sampling-ratio>0.75</sampling-ratio>
  
    <beta>0.5</beta>
     
    <dataset>
      <train-dataset>data/Yeast/Yeast-train1.arff</train-dataset>
      <test-dataset>data/Yeast/Yeast-test1.arff</test-dataset>
      <xml>data/Yeast/Yeast.xml</xml>
    </dataset>
    
    <listener type="g3pkemlc.Listener">
      <report-dir-name>reports/G3P-kEMLC</report-dir-name>
      <global-report-name>summaryEnsembleMLC</global-report-name>
      <report-frequency>1</report-frequency> 
    </listener>
  </process>
</experiment>

```

* The configuration file must start with the ```<experiment>``` tag and then the ```<process>``` tag, the last indicating the class with the evolutionary algorithm, in our case ```g3pkemlc.Alg```.
* The ```<rand-gen-factory>``` must determine the seed for random numbers with the ```seed``` attribute. If several seeds are going to be used, the tag ```<rand-gen-factory multi="true">``` may be used, including inside the different seeds, as follows:
  ```xml
    <rand-gen-factory multi="true">
	  <rand-gen-factory seed="10"/>
	  <rand-gen-factory seed="20"/>
	  <rand-gen-factory seed="30"/>
	    ...
    </rand-gen-factory>
  ```
* The size of the population is determined with the ```<population-size>``` tag.
* The number of generations of the evolutionary algorithm is determined with the ```<max-of-generations>``` tag.
* The ```<recombinator>``` tag determines with the ```rec-prob``` attribute, the probability of the recombinator or crossover operator.
* The ```<mutator>``` tag determines with the ```mut-prob``` attribute, the probability of the mutation operator.
* The size of the _k_-labelsets is determined with the ```<min-k>``` and ```<max-k>``` tags. If fixed _k_ is desired for all classifiers, both values are the same.
* The maximum allowed depth of the tree is determined with the ```<max-depth>``` tag.
* The maximum number of children of each node of the tree is determined with the ```<max-children>``` tag.
* The number of classifier in the pool is determined with the ```<v>``` tag. It determines the average number of votes per label that are expected in the pool of classifiers, and depending on the value of k, it automatically calculates the number of classifiers to build.
* The ratio of instances to sampe (whithout replacement) at each base classifier is determined with the ```<sampling-ratio>``` tag.
* The value of beta to combine the terms of the fitness function is determined with the ```<beta>``` tag.
* With the ```<dataset>``` tag, the datasets used for training (for the G3P algorithm) and testing (for testing the final ensemble obtained by G3P-kEMLC) are determined with the tags ```<train-dataset>``` and ```<test-dataset>``` respectively. The ```<xml>``` tag indicates the xml file of the dataset (Mulan format, [see more](http://www.uco.es/kdis/mllresources/#MulanFormat)).  Several datasets, or several partitions of the same dataset may be used, including the tag ```<dataset multi="true">```, and the different datasets inside, as follows:
  ```xml
    <dataset multi="true">
      <dataset>
        <train-dataset>data/Yeast-train1.arff</train-dataset>
        <test-dataset>data/Yeast-test1.arff</test-dataset>
        <xml>data/Yeast.xml</xml>
      </dataset>
      <dataset>
        <train-dataset>data/Yeast-train2.arff</train-dataset>
        <test-dataset>data/Yeast-test2.arff</test-dataset>
        <xml>data/Yeast.xml</xml>
      </dataset>
      <dataset>
        <train-dataset>data/Yeast-train3.arff</train-dataset>
        <test-dataset>data/Yeast-test3.arff</test-dataset>
        <xml>data/Yeast.xml</xml>
      </dataset>
      </dataset>
        ...
    </dataset>
  ```
* The ```<listener>``` tag determines the class used as listener; it is the responsible of creating the different reports during and at the end of the evolutionary process. By default, the listener used is the one of the ```g3pkemlc.Listener``` class. The ```<report-dir-name>``` tag determines the directory where the reports of the different executions are stored. The ```<global-report-name>``` tag indicates the filename of the global report file. Finally, the ```<report-frequency>``` tag indicates the frequency with which the reports for the iterations are created.

Then, several more characteristics of the evolutionary algorithm could be modified in the configuration file, but they are optional and default values for them are given if they are not included in this file:
* The parents selector is determined with the ```<parents-selector>``` tag. By default, tournament selection of size 2 is used. In order to change the size of the tournament selection, the sub-tag ```<tournament-size>``` could be used.
* By default, internal nodes in the tree combine the predictions as bipartitions of their children. However, confidences could be used instead of bipartitions at all nodes with the tag ```<use-confidences>``` to _true_ value.

*Yeast* [[Eli01]](#Eli01) multi-label dataset has been included in the repository as example; however, a wide variety of dataset are available at the [KDIS Research Group Repository](http://www.uco.es/kdis/mllresources/). Further, the example configuration file (*Yeast.cfg*) is also provided.

### References
<a name="Eli01"></a>**[Eli01]** A. Elisseeff and J. Weston. (2001). A kernel method for multi-labelled classification. In Advances in Neural Information Processing Systems, 14, 681–687.

<a name="Hal09"></a>**[Hal09]** M. Hall, E. Frank, G. Holmes, B. Pfahringer, P. Reutemann, and I. H. Witten. (2009). The WEKA data mining software: an update. ACM SIGKDD explorations newsletter, 11(1), 10-18.

<a name="Tso11"></a>**[Tso11]** G. Tsoumakas, E. Spyromitros-Xioufis, J. Vilcek, and I. Vlahavas. (2011). Mulan: A java library for multi-label learning. Journal of Machine Learning Research, 12, 2411-2414.

<a name="Ven08"></a>**[Ven08]** S. Ventura, C. Romero, A. Zafra, J. A. Delgado, and C. Hervás. (2008). JCLEC: a Java framework for evolutionary computation. Soft Computing, 12(4), 381-392.