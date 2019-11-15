package gpemlc;

import java.util.ArrayList;
import java.util.Hashtable;

import org.apache.commons.configuration.Configuration;

import gpemlc.mutator.Mutator;
import gpemlc.recombinator.Crossover;
import mulan.classifier.InvalidDataException;
import mulan.classifier.MultiLabelLearner;
import mulan.classifier.MultiLabelOutput;
import mulan.classifier.transformation.ClassifierChain;
import mulan.classifier.transformation.LabelPowerset2;
import mulan.core.MulanException;
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
import weka.core.Instances;

public class Alg extends SGE {

	/**
	 * 
	 */
	private static final long serialVersionUID = -790335501425435317L;

	BettersSelector bselector = new BettersSelector(this);
	
	/**
	 * Max number of children at each node
	 */
	int maxChildren = 5;
	
	/**
	 * Max depth of the tree
	 */
	int maxDepth = 2;
	
	/**
	 * Full training dataset
	 */
	MultiLabelInstances fullTrainData;
	
	/**
	 * Training datasets
	 */
	MultiLabelInstances [] trainData;
	
	/**
	 * Ratio of instances sampled at each train data
	 */
	double sampleRatio;
	
	/**
	 * Test dataset
	 */
	MultiLabelInstances testData;
	
	/**
	 * Number of MLC
	 */
	int nMLC = 10;
	
	/**
	 * Built classifiers
	 */
	Hashtable<String, MultiLabelLearner> table;
	
	Hashtable<String, Prediction> tablePredictions;
	
	Hashtable<String, String> tableChains;
	 
	IRandGen randgen;
	
	@Override
	public void configure(Configuration configuration) {
		super.configure(configuration);
		
		table = new Hashtable<String, MultiLabelLearner>();
		tablePredictions = new Hashtable<String, Prediction>();
		
		String datasetTrainFileName = configuration.getString("dataset.train-dataset");
		String datasetTestFileName = configuration.getString("dataset.test-dataset");
		String datasetXMLFileName = configuration.getString("dataset.xml");
		
		sampleRatio = configuration.getDouble("sampling-ratio");
		
		randgen = randGenFactory.createRandGen();
		
		fullTrainData = null;
		testData = null;
		try {
			fullTrainData = new MultiLabelInstances(datasetTrainFileName, datasetXMLFileName);
			Instances evalData = fullTrainData.getDataSet();
			testData = new MultiLabelInstances(datasetTestFileName, datasetXMLFileName);
			
			trainData = new MultiLabelInstances[nMLC];
			for(int p=0; p<nMLC; p++) {
				trainData[p] = MulanUtils.sampleData(fullTrainData, sampleRatio, randgen);
				
//				LabelPowerset2 lp = new LabelPowerset2(new J48());
//				lp.setSeed(1);
//				table.put(String.valueOf(p), lp);
//				table.get(String.valueOf(p)).build(trainData[p]);
				
				table.put(String.valueOf(p), new ClassifierChain(new J48(), randomChain(fullTrainData.getNumLabels())));
				table.get(String.valueOf(p)).build(trainData[p]);
				
				Prediction pred = new Prediction(fullTrainData.getNumInstances(), fullTrainData.getNumLabels());
				for(int i=0; i<fullTrainData.getNumInstances(); i++) {
					boolean[] bip = table.get(String.valueOf(p)).makePrediction(fullTrainData.getDataSet().get(i)).getBipartition();
					for(int j=0; j<fullTrainData.getNumLabels(); j++) {
						if(bip[j]) {
							pred.bip[i][j] = 1;
						}
						else {
							pred.bip[i][j] = 0;
						}
					}
				}
				
				tablePredictions.put(String.valueOf(p), pred);
				
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		
		((StringTreeCreator)provider).setMaxChildren(maxChildren);
		((StringTreeCreator)provider).setMaxDepth(maxDepth);
		((StringTreeCreator)provider).setnMax(nMLC);
		
		((Mutator)mutator.getDecorated()).setMaxTreeDepth(maxDepth);
		((Mutator)mutator.getDecorated()).setnChilds(maxChildren);
		((Mutator)mutator.getDecorated()).setnMax(nMLC);
		
		((Crossover)recombinator.getDecorated()).setMaxTreeDepth(maxDepth);
		
//		((Evaluator)evaluator).setClassifiers(classifiers);
		((Evaluator)evaluator).setFullTrainData(fullTrainData);
		((Evaluator)evaluator).setTablePredictions(tablePredictions);
	}
	
	@Override
	protected void doInit() {
		super.doInit();
		
	}
	
	int [] randomChain(int n) {
		int[] chain = new int[n];
		for(int i=0; i<n; i++) {
			chain[i] = i;
		}
		
		int r, aux;
		for(int i=0; i<n; i++) {
			r = randgen.choose(n);
			aux = chain[r];
			chain[r] = chain[i];
			chain[i] = aux;
		}
		
		return chain;
	}
	
	protected void doControl()
	{
//		System.out.println("Generation " + generation);
		
		if (generation >= maxOfGenerations) {
			EMLC ensemble = new EMLC(new ClassifierChain(new J48()));
			String bestGenotype = ((StringTreeIndividual)bselector.select(bset, 1).get(0)).getGenotype();
			ensemble.setGenotype(bestGenotype);
			ensemble.setTable(table);
			
			try {
				ensemble.build(fullTrainData);
				
				ArrayList<Measure> measures = new ArrayList<Measure>();
				measures.add(new ExampleBasedFMeasure());
				Evaluation results = new Evaluation(measures, testData);
				mulan.evaluation.Evaluator eval = new mulan.evaluation.Evaluator();
				results = eval.evaluate(ensemble, testData);
				System.out.println(results);
				
				
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			
			state = FINISHED;
			return;
		}
	}	
	
}
