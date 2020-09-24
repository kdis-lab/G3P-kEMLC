# AG3P-kEMLC: Auto-adaptive G3P algorithm to build Ensembles of Multi-Label Classifiers

AG3P-kEMLC is an auto-adaptive Grammar-Guided Genetic Programming (G3P) algorithm designed to build Ensembles of Multi-Label Classifiers (EMLCs) in a tree-shape.
The method auto-adapts some of the parameters of the G3P algorithm (such as the crossover and mutation probabilities) in order to reduce the selection of hyperparameters and to adapt to each specific case. Besides, it has a stop condition based on the non-improvement in fitness of the best individual to avoid a high runtime and to prevent overfitting.
Given the use of G3P, a tree-shaped ensemble is obtained, where at each node of the tree the prediction of children nodes are combined, and the leaves are the base multi-label classifiers.
Each of the base classifiers is focused on a subset of the labels of size _k_ (a.k.a. _k_-labelset). Unlike in other EMLCs, G3P-kEMLC is able to deal with base classifiers focused on _k_-labelsets of different size. Three different options to select the size of the _k_-labelsets are provided: 1) fixed value of _k_; 2) variable value of _k_ between _kMin_ and _kMax_, where all values have the same probability; and 3) variable value of _k_ between _kMin_ and _kMax_, where smaller _k_ values have more probability to be selected than higher values.
More information about this algorithm will be provided soon.

In this repository we provide the code of AG3P-kEMLC, distributed under the GPLv3 License. AG3P-kEMLC has been implemented using JCLEC [[Ven08]](#Ven08), Mulan [[Tso11]](#Tso11), and Weka [[Hal09]](#Hal09) libraries. Besides, the latest release [(v 3.0)](https://github.com/kdis-lab/G3P-kEMLC/releases/tag/v3.0) provides the executable jar to execute AG3P-kEMLC.
The code and executable of a previous version, presented in IEEE-CEC [Moy20](#Moy20) is available in the [v 2.0 release](https://github.com/kdis-lab/G3P-kEMLC/releases/tag/v2.0).

To execute AG3P-kEMLC, the following command should be executed:
```sh
java -jar AG3P-kEMLC.jar configFile.xml
```

The configuration file is a xml file including the parameters of the G3P algorithm, and following is presented an example. There are few parameters that are mandatory, while the rest are set a default value if they are not indicated.

```xml
<experiment>
  <process algorithm-type="g3pkemlc.Alg">
    <rand-gen-factory seed="10"/>
     
    <population-size>50</population-size>
    <max-of-generations>200</max-of-generations>    
          
    <dataset>
      <train-dataset>data/Yeast/Yeast-train1.arff</train-dataset>
      <test-dataset>data/Yeast/Yeast-test1.arff</test-dataset>
      <xml>data/Yeast/Yeast.xml</xml>
    </dataset>
    
    <listener type="g3pkemlc.Listener">
      <report-frequency>1</report-frequency> 
    </listener>

    <!-- Optional parameters. If not included, they are set to the default values presented. -->

    <recombinator rec-prob="0.5" />
    <mutator mut-prob="0.5" />
     
    <min-k>3</min-k>
    <max-k>7</max-k> <!-- numLabels / 2 -->
    <k-mode>gaussian</k-mode>

    <max-depth>3</max-depth>
    <max-children>7</max-children>
    
    <v>20</v>

    <sampling-ratio>0.75</sampling-ratio>
  
    <beta>0.5</beta>

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
* The maximum number of generations of the evolutionary algorithm is determined with the ```<max-of-generations>``` tag. Although the algorithm has a stop condition based on the quality of the best individual, a maximum number of generations is still needed.
* With the ```<dataset>``` tag, the datasets used for training (for the G3P algorithm) and testing (for testing the final ensemble obtained by AG3P-kEMLC) are determined with the tags ```<train-dataset>``` and ```<test-dataset>``` respectively. The ```<xml>``` tag indicates the xml file of the dataset (Mulan format, [see more](http://www.uco.es/kdis/mllresources/#MulanFormat)).  Several datasets, or several partitions of the same dataset may be used, including the tag ```<dataset multi="true">```, and the different datasets inside, as follows:
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
* The ```<listener>``` tag determines the class used as listener; it is the responsible of creating the different reports during and at the end of the evolutionary process. By default, the listener used is the one of the ```g3pkemlc.Listener``` class. The ```<report-frequency>``` tag indicates the frequency or number of generations between each iteration report is created.
* The ```<recombinator>``` tag determines with the ```rec-prob``` attribute, the initial probability of the recombinator or crossover operator. By default, it is 0.5 (as well as the mutation probability), and it will adapt during the evolution.
* The ```<mutator>``` tag determines with the ```mut-prob``` attribute, the initial probability of the mutation operator. By default, it is 0.5 (as well as the crossover probability), and it will adapt during the evolution.
* The size of the _k_-labelsets is determined with the ```<min-k>``` and ```<max-k>``` tags. If fixed _k_ is desired for all classifiers, both values are the same. By default, ```<min-k>``` is set to 3, and ```<max-k>``` to half the number of labels.
* The ```<k-mode>``` tag determines the way in which the size of each _k_-labelset is selected. If _uniform_ mode is selected, all values of _k_ between _min-k_ and _max-k_ have the same probability to be selected at each classifier. On the other hand, if _gaussian_ mode is selected (by default), higher values of _k_ in the predefined range have lower probability to be selected, where the probability for higher values decreases as a gaussian function. This second mode would lead to create a higher number of smaller _k_-labelsets.
* The maximum allowed depth of the tree is determined with the ```<max-depth>``` tag. By default, it is set to 3.
* The maximum number of children of each node of the tree is determined with the ```<max-children>``` tag. By default it is set to 7.
* The number of classifier in the pool is determined with the ```<v>``` tag. It determines the average number of votes per label that are expected in the pool of classifiers, and depending on the value of k, it automatically calculates the number of classifiers to build. By default, 20 votes per label are expected in average in the initial pool
* The ratio of instances to sample (whithout replacement) at each base classifier is determined with the ```<sampling-ratio>``` tag. By default, 75% of instances are sampled for each classifier.
* The value of beta to combine the terms of the fitness function is determined with the ```<beta>``` tag. By default, 0.5 is used, giving the same value to both metrics in fitness.

Then, several more characteristics of the evolutionary algorithm could be modified in the configuration file, but they are just optional and default values for them are given if they are not included in this file:
* The parents selector is determined with the ```<parents-selector>``` tag. By default, tournament selection of size 2 is used. In order to change the size of the tournament selection, the sub-tag ```<tournament-size>``` could be used.

*Emotions* [[Tso08]](#Tso08) and *Yeast* [[Eli01]](#Eli01) multi-label datasets have been included in the repository as example; however, a wide variety of dataset are available at the [KDIS Research Group Repository](http://www.uco.es/kdis/mllresources/). Further, the example configuration files (*Emotions.xml* and *Yeast.xml*) are also provided.

### References
<a name="Eli01"></a>**[Eli01]** A. Elisseeff and J. Weston. (2001). A kernel method for multi-labelled classification. In _Advances in Neural Information Processing Systems_, 14, 681–687.

<a name="Hal09"></a>**[Hal09]** M. Hall, E. Frank, G. Holmes, B. Pfahringer, P. Reutemann, and I. H. Witten. (2009). The WEKA data mining software: an update. _ACM SIGKDD explorations newsletter_, 11(1), 10-18.

<a name="Moy20"></a>**[Moy20]** J.M. Moyano, E. Gibaja, K. Cios, S. Ventura. (2020). Tree-shaped ensemble of multi-label classifiers using grammar-guided genetic programming, In _2020 IEEE Congress on Evolutionary Computation, CEC 2020_. 1-8.

<a name="Tso08"></a>**[Tso08]** G. Tsoumakas, I. Katakis, and I. Vlahavas. (2008). Effective and Efficient Multilabel Classification in Domains with Large Number of Labels. In _ECML/PKDD 2008 Workshop on Mining Multidimensional Data (MMD’08)_, 53-59.

<a name="Tso11"></a>**[Tso11]** G. Tsoumakas, E. Spyromitros-Xioufis, J. Vilcek, and I. Vlahavas. (2011). Mulan: A java library for multi-label learning. _Journal of Machine Learning Research_, 12, 2411-2414.

<a name="Ven08"></a>**[Ven08]** S. Ventura, C. Romero, A. Zafra, J. A. Delgado, and C. Hervás. (2008). JCLEC: a Java framework for evolutionary computation. _Soft Computing_, 12(4), 381-392.