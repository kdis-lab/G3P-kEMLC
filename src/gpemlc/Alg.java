package gpemlc;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Hashtable;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

import org.apache.commons.configuration.Configuration;

import gpemlc.utils.DatasetTransformation;
import gpemlc.utils.KLabelset;
import gpemlc.utils.KLabelsetGenerator;
import gpemlc.utils.MulanUtils;
import gpemlc.utils.TreeUtils;
import gpemlc.utils.Utils;
import gpemlc.mutator.Mutator;
import gpemlc.recombinator.Crossover;
import mulan.classifier.MultiLabelLearnerBase;
import mulan.classifier.transformation.LabelPowerset2;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluation;
import mulan.evaluation.measure.ExampleBasedFMeasure;
import mulan.evaluation.measure.Measure;
import net.sf.jclec.IIndividual;
import net.sf.jclec.algorithm.classic.SGE;
import net.sf.jclec.fitness.SimpleValueFitness;
import net.sf.jclec.selector.BettersSelector;
import net.sf.jclec.selector.WorsesSelector;
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
	 * Minimum size of the k-labelsets
	 */
	int minK;
	
	/**
	 * Maximum size of the k-labelsets
	 */
	int maxK;
	
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
	 * Array of k-labelsets
	 */
	ArrayList<KLabelset> klabelsets;
	
	/**
	 * Final ensemble
	 */
	EMLC ensemble;
	
	/**
	 * Lock for parallel critical code
	 */
	Lock lock = new ReentrantLock();
	
	/** Betters selector. Used in update phase */
	BettersSelector bettersSelector = new BettersSelector(this);
	
	/** Worses selector. Used in update phase */
	WorsesSelector worsesSelector = new WorsesSelector(this);
	
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
		
		minK = configuration.getInt("min-k");
		maxK = configuration.getInt("max-k");
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
			
			//Generate k-labelsets
			KLabelsetGenerator klabelsetGen = new KLabelsetGenerator(minK, maxK, fullTrainData.getNumLabels(), nMLC);
			klabelsetGen.setRandgen(randgen);
			klabelsets = klabelsetGen.generateKLabelsets();
			klabelsetGen.printKLabelsets();
			
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
		
//		System.out.println("gen: " + generation);
		
		if (generation >= maxOfGenerations) {			
//			System.exit(1);
			//Get base learner
//			learner = null;
			learner = new LabelPowerset2(new J48());
			((LabelPowerset2)learner).setSeed(seed);
			
			//Generate ensemble object
			ensemble = new EMLC(learner, klabelsets, bestGenotype, useConfidences);
			
			System.out.println(Arrays.toString(TreeUtils.votesPerLabel(bestGenotype, klabelsets, fullTrainData.getNumLabels())));
			
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
	 * Clear some variables to null
	 */
	public void clear() {
		fullTrainData = null;
		currentTrainData = null;
		tablePredictions = null;
		klabelsets = null;
		testData = null;
		learner = null;
		ensemble = null;
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
		IRandGen randgen = null;
		MultiLabelInstances currentTrainData, currentFullData;
		MultiLabelLearnerBase learner;
		int seed = c;
		
		try {
			randgen = new RanecuFactory2().createRandGen(seed, seed*2);
			
			//Sample c-th data
			currentTrainData = MulanUtils.sampleData(fullTrainData, sampleRatio, randgen);
			
			//Build classifier with c-th data
			learner = null;
			learner = new LabelPowerset2(new J48());
			((LabelPowerset2)learner).setSeed(seed);
			//Transform full train data
			DatasetTransformation dt = new DatasetTransformation();
			currentTrainData = dt.transformDataset(currentTrainData, klabelsets.get(c).getKlabelset());
			if(sampleRatio >= 0.999) {
				currentFullData = currentTrainData;
			}
			else {
				currentFullData = dt.transformDataset(fullTrainData, klabelsets.get(c).getKlabelset());
			}
			//Build
			learner.build(currentTrainData);
			
			//Store object of classifier in the hard disk				
			utils.writeObject(learner, "mlc/classifier"+c+".mlc");
			
			//Get predictions of c-th classifier over all data
			double[][] currentPredictions = new double[currentFullData.getNumInstances()][klabelsets.get(c).k];
			for(int i=0; i<currentFullData.getNumInstances(); i++) {
				if(useConfidences) {
					System.arraycopy(learner.makePrediction(currentFullData.getDataSet().get(i)).getConfidences(), 0, currentPredictions[i], 0, currentFullData.getNumLabels());
				}
				else {
					System.arraycopy(utils.bipartitionToConfidence(learner.makePrediction(currentFullData.getDataSet().get(i)).getBipartition()), 0, currentPredictions[i], 0, currentFullData.getNumLabels());
				}
			}
			
			//Create Prediction object
			Prediction pred = new Prediction(dt.getOriginalLabelIndices(), currentPredictions);
			
			
			//Put predictions in table
			lock.lock();
			tablePredictions.put(String.valueOf(c), new Prediction(pred));
			lock.unlock();
			
			currentTrainData = null;
			currentFullData = null;
			pred = null;
			learner = null;
			currentPredictions = null;
			dt = null;
		} catch(Exception e) {
			e.printStackTrace();
			System.exit(-1);
		}		
	}
	
//	protected void doUpdate() 
//	{
//		IIndividual bestb = bettersSelector.select(bset, 1).get(0);
//		IIndividual bestc = bettersSelector.select(cset, 1).get(0);
//		
//		ArrayList<IIndividual> toRemove = new ArrayList<IIndividual>();
//		for(IIndividual ind : cset) {
//			if(((SimpleValueFitness)ind.getFitness()).getValue() < 0) {
//				toRemove.add(ind);
//			}
//		}
////		System.out.println("remove: " + toRemove.size());
//		for(IIndividual ind : toRemove) {
//			cset.remove(ind);
//		}
//		
//		int csetSpace = populationSize - cset.size();
//		if(csetSpace <= 0) {
//			if (evaluator.getComparator().compare(bestb.getFitness(), bestc.getFitness()) == 1) {
//				IIndividual worstc = worsesSelector.select(cset, 1).get(0);
//				cset.remove(worstc);
//				cset.add(bestb);
//			}
//		}
//		else {
//			bset = bettersSelector.select(bset, bset.size());
//			for(int i=0; i<csetSpace; i++) {
//				cset.add(bset.get(i));
//			}
//		}
////		
////		for(int i=0; i<nBest; i++) {
////			if (evaluator.getComparator().compare(bestb.get(i).getFitness(), bestc.getFitness()) == 1) {
////				IIndividual worstc = worsesSelector.select(cset, 1).get(0);
////				cset.remove(worstc);
////				cset.add(bestb.get(i));
////			}
////			else {
////				break;
////			}
////		}
////		
////		IIndividual rInd;
////		List<IIndividual> toRemove = new ArrayList<IIndividual>();
////		List<IIndividual> toAdd = new ArrayList<IIndividual>();
////		for(IIndividual ind : cset) {
////			if(((SimpleValueFitness)ind.getFitness()).getValue() < 0) {
////				do {
////					rInd = bset.get(randgen.choose(0, bset.size()));
////				}while(((SimpleValueFitness)rInd.getFitness()).getValue() < 0);
////				toRemove.add(ind);
////				toAdd.add(rInd);
////			}
////		}
////		
////		for(IIndividual ind : toRemove) {
////			if(cset.contains(ind)) {
////				cset.remove(ind);
////			}
////		}
////		
////		for(IIndividual ind : toAdd) {
////			if(cset.contains(ind)) {
////				cset.add(ind);
////			}
////		}
////		
////		while(cset.size() < populationSize) {
////			cset.add(bset.get(randgen.choose(0, bset.size())));
////		}
////		
//		// Sets new bset
//		bset = cset;
//		// Clear pset, rset & cset
//		pset = null;
//		rset = null;
//		cset = null;
//		
//	}
	
}
