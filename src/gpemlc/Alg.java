package gpemlc;

import java.io.File;
import java.util.ArrayList;
import java.util.Hashtable;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

import org.apache.commons.configuration.Configuration;

import gpemlc.Utils.ClassifierType;
import gpemlc.mutator.Mutator;
import gpemlc.recombinator.Crossover;
import mulan.classifier.MultiLabelLearnerBase;
import mulan.classifier.transformation.BinaryRelevance;
import mulan.classifier.transformation.ClassifierChain;
import mulan.classifier.transformation.LabelPowerset2;
import mulan.classifier.transformation.PrunedSets2;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluation;
import mulan.evaluation.measure.ExampleBasedFMeasure;
import mulan.evaluation.measure.Measure;
import net.sf.jclec.algorithm.classic.SGE;
import net.sf.jclec.selector.BettersSelector;
import net.sf.jclec.stringtree.StringTreeCreator;
import net.sf.jclec.stringtree.StringTreeIndividual;
import net.sf.jclec.util.random.IRandGen;
import weka.classifiers.trees.J48;

/**
 * Class implementing the evolutionary algorithm.
 * 
 * @author Jose M. Moyano
 *
 */
public class Alg extends SGE {

	/**
	 * serialVersionUID
	 */
	private static final long serialVersionUID = -790335501425435317L;

	/**
	 * Betters selector
	 */
	BettersSelector bselector = new BettersSelector(this);
	
	/**
	 * Max number of children at each node
	 */
	int maxChildren;
	
	/**
	 * Max depth of the tree
	 */
	int maxDepth;
	
	/**
	 * Full training dataset
	 */
	MultiLabelInstances fullTrainData;
	
	/**
	 * Current sampled training data for a given multi-label classifier
	 */
	MultiLabelInstances currentTrainData;
	
	/**
	 * Ratio of instances sampled at each train data
	 */
	double sampleRatio;
	
	/**
	 * Test dataset
	 */
	MultiLabelInstances testData;
	
	/**
	 * Number of different MLC
	 */
	int nMLC;
	
	/**
	 * Utils
	 */
	Utils utils;

	/**
	 * Table with predictions of each classifier
	 */
	Hashtable<String, Prediction> tablePredictions;
	
	/**
	 * Random numbers generator
	 */
	IRandGen randgen;
	
	/**
	 * Multi-label base classifier
	 */
	MultiLabelLearnerBase learner;
	
	/**
	 * Seed for random numbers
	 */
	int seed;
	
	/**
	 * Use confidences or bipartitions in combining predictions
	 */
	boolean useConfidences;
	
	/**
	 * Type of classifier used: LP, CC, k-labelset
	 */
	Utils.ClassifierType classifierType;
	
	/**
	 * Final ensemble
	 */
	EMLC ensemble;
	
	/**
	 * Lock for parallel critical code
	 */
	Lock lock = new ReentrantLock();
	
	/**
	 * Getter for test data
	 * 
	 * @return Test data
	 */
	public MultiLabelInstances getTestData() {
		return testData;
	}
	
	/**
	 * Getter for the seed
	 * 
	 * @return Seed
 	 */
	public int getSeed() {
		return seed;
	}
	
	/**
	 * Getter for classifierType
	 * 
	 * @return classifierType
	 */
	public Utils.ClassifierType getClassifierType() {
		return classifierType;
	}
	
	/**
	 * Getter for the ensemble
	 * 
	 * @return Ensemble
	 */
	public EMLC getEnsemble() {
		return ensemble;
	}
	
	@Override
	public void configure(Configuration configuration) {
		super.configure(configuration);
		
		seed = configuration.getInt("rand-gen-factory[@seed]");
		randgen = randGenFactory.createRandGen();
		utils = new Utils(randgen);
		
		
		//Initialize table for predictions
		tablePredictions = new Hashtable<String, Prediction>();
		
		//Get datasets
		String datasetTrainFileName = configuration.getString("dataset.train-dataset");
		String datasetTestFileName = configuration.getString("dataset.test-dataset");
		String datasetXMLFileName = configuration.getString("dataset.xml");
		
		sampleRatio = configuration.getDouble("sampling-ratio");
		
		nMLC = configuration.getInt("different-classifiers");
		maxChildren = configuration.getInt("max-children");
		maxDepth = configuration.getInt("max-depth");
		useConfidences = configuration.getBoolean("use-confidences");
		
		fullTrainData = null;
		currentTrainData = null;
		testData = null;
		try {
			fullTrainData = new MultiLabelInstances(datasetTrainFileName, datasetXMLFileName);
			testData = new MultiLabelInstances(datasetTestFileName, datasetXMLFileName);

			//Create folder for classifiers if it does not exist
			File f = new File("mlc/");
			if (!f.exists()) {
			   f.mkdir();
			}
			
			String learnerType = configuration.getString("base-learner");
			switch (learnerType.toUpperCase()) {
			case "BR":
				classifierType = ClassifierType.BR;
				break;
			
			case "LP":
				classifierType = ClassifierType.LP;
				break;
			
			case "CC":
				classifierType = ClassifierType.CC;
				break;
				
			case "PS":
				classifierType = ClassifierType.PS;
				break;
				
			case "KLABELSET":
			case "K-LABELSET":
				classifierType = ClassifierType.kLabelset;
				break;

			default:
				classifierType = null;
				break;
			}
			
			//Set number of threads
			ExecutorService executorService = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
			
			//Create, store, and get predictions of each different classifier
			for(int c=0; c<nMLC; c++) {
				executorService.execute(new BuildClassifierParallel(c));				
			}
			executorService.shutdown();
			
			try {
				//Wait until all threads finish
				executorService.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
			} catch (Exception e) {
				e.printStackTrace();
			}
			
			currentTrainData = null;
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		//Set settings of provider, genetic operators and evaluator
		((StringTreeCreator)provider).setMaxChildren(maxChildren);
		((StringTreeCreator)provider).setMaxDepth(maxDepth);
		((StringTreeCreator)provider).setnMax(nMLC);
		
		((Mutator)mutator.getDecorated()).setMaxTreeDepth(maxDepth);
		((Mutator)mutator.getDecorated()).setnChilds(maxChildren);
		((Mutator)mutator.getDecorated()).setnMax(nMLC);
		
		((Crossover)recombinator.getDecorated()).setMaxTreeDepth(maxDepth);
		
		((Evaluator)evaluator).setFullTrainData(fullTrainData);
		((Evaluator)evaluator).setTablePredictions(tablePredictions);
		((Evaluator)evaluator).setUseConfidences(useConfidences);
		((Evaluator)evaluator).setRandgen(randgen);
	}
	
	@Override
	protected void doInit() {
		super.doInit();
	}
	
	@Override
	protected void doControl()
	{		
		//Get genotype of best individual
		String bestGenotype = ((StringTreeIndividual)bselector.select(bset, 1).get(0)).getGenotype();
		
		if (generation >= maxOfGenerations) {
			//Get base learner
			switch (classifierType) {
			case BR:
				learner = new BinaryRelevance(new J48());
				break;
				
			case LP:
				learner = new LabelPowerset2(new J48());
				((LabelPowerset2)learner).setSeed(seed);
				break;
			
			case CC:
				learner = new ClassifierChain(new J48(), utils.randomPermutation(fullTrainData.getNumLabels(), randgen));
				break;
				
			case PS:
				learner = new PrunedSets2();
				break;
				
			case kLabelset:
				learner = null;
				break;

			default:
				learner = null;
				break;
			}
			
			//Generate ensemble object
			ensemble = new EMLC(learner, bestGenotype, useConfidences);
			
			//Print the leaves; i.e., different classifiers used in the ensemble
			System.out.println(utils.getLeaves(bestGenotype));
			
			try {
				//Build the ensemble
				ensemble.build(fullTrainData);
				
				//After building the ensemble, we can remove all the classifiers built and stored in hard disk
				utils.purgeDirectory(new File("mlc/"));				
			} catch (Exception e) {
				e.printStackTrace();
			}
			
			state = FINISHED;
			return;
		}
	}	
	
	/**
	 * Class to parallelize building base classifiers
	 * 
	 * @author Jose M. Moyano
	 *
	 */
	public class BuildClassifierParallel extends Thread {
		
		//Index of classifier to build
		int index;
		
		public BuildClassifierParallel(int index) {
			this.index = index;
		}
		
		public void run() {
			try {
				buildClassifier(index);
			}catch(Exception e) {
				e.printStackTrace();
				System.exit(-1);
			}
		}
	}
	
	/**
	 * Build the c-th classifier
	 * 
	 * @param c Index of classifier to build
	 */
	public void buildClassifier(int c) {
		IRandGen randgen;
		MultiLabelInstances currentTrainData;
		MultiLabelLearnerBase learner;
		int seed = c;
		
		try {
//			randgen = randGenFactory.createRandGen();
			randgen = new RanecuFactory2().createRandGen(seed, seed*2);
			
			//Sample c-th data
			currentTrainData = MulanUtils.sampleData(fullTrainData, sampleRatio, randgen);
			
			//Build classifier with c-th data
			learner = null;
			switch (classifierType) {
			case BR:
				learner = new BinaryRelevance(new J48());
				break;
			
			case LP:
				learner = new LabelPowerset2(new J48());
				((LabelPowerset2)learner).setSeed(seed);
				break;
			
			case CC:
				learner = new ClassifierChain(new J48(), utils.randomPermutation(fullTrainData.getNumLabels(), randgen));
				break;
				
			case PS:
				learner = new PrunedSets2();
				break;
				
			case kLabelset:
				learner = null;
				break;

			default:
				classifierType = null;
				learner = null;
				break;
			}
			
			learner.build(currentTrainData);
			
			//Store object of classifier in the hard disk				
			utils.writeObject(learner, "mlc/classifier"+c+".mlc");
			
			//Get predictions of c-th classifier over all data
			Prediction pred = new Prediction(fullTrainData.getNumInstances(), fullTrainData.getNumLabels());
			for(int i=0; i<fullTrainData.getNumInstances(); i++) {
				if(useConfidences) {
					System.arraycopy(learner.makePrediction(fullTrainData.getDataSet().get(i)).getConfidences(), 0, pred.pred[i], 0, fullTrainData.getNumLabels());
				}
				else {
					System.arraycopy(utils.bipartitionToConfidence(learner.makePrediction(fullTrainData.getDataSet().get(i)).getBipartition()), 0, pred.pred[i], 0, fullTrainData.getNumLabels());
				}
			}
			
			
			//Put predictions in table
			lock.lock();
			tablePredictions.put(String.valueOf(c), pred);
			lock.unlock();
		} catch(Exception e) {
			e.printStackTrace();
			System.exit(-1);
		}
		
		
	}
	
	public double evalBestTest() {
		double testExF = 0.0;
		EMLC ensemble;
		
		String bestGenotype = ((StringTreeIndividual)bselector.select(bset, 1).get(0)).getGenotype();
		
		//Get base learner
		switch (classifierType) {
		case BR:
			learner = new BinaryRelevance(new J48());
			break;
			
		case LP:
			learner = new LabelPowerset2(new J48());
			((LabelPowerset2)learner).setSeed(seed);
			break;
		
		case CC:
			learner = new ClassifierChain(new J48(), utils.randomPermutation(fullTrainData.getNumLabels(), randgen));
			break;
			
		case PS:
			learner = new PrunedSets2();
			break;
			
		case kLabelset:
			learner = null;
			break;

		default:
			learner = null;
			break;
		}
		
		//Generate ensemble object
		ensemble = new EMLC(learner, bestGenotype, useConfidences);
		
		//Print tree
//		System.out.println(bestGenotype);
		
		try {
			//Build the ensemble
			ensemble.build(fullTrainData);

			Evaluation results;
			
			List<Measure> measures = new ArrayList<Measure>(1);
			measures.add(new ExampleBasedFMeasure());
			results = new Evaluation(measures, testData);
			mulan.evaluation.Evaluator eval = new mulan.evaluation.Evaluator();
			
			if(classifierType == ClassifierType.LP || classifierType == ClassifierType.PS) {
				ensemble.resetSeed(seed);
			}
			results = eval.evaluate(ensemble, testData, measures);
			
			testExF = results.getMeasures().get(0).getValue();
			
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		return testExF;
	}
	
}
