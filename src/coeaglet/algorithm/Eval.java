package coeaglet.algorithm;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.Hashtable;
import java.util.List;

import coeaglet.utils.DatasetTransformation;
import mulan.classifier.MultiLabelLearner;
import mulan.classifier.transformation.LabelPowerset2;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluation;
import mulan.evaluation.MulanLP2Evaluator;
import mulan.evaluation.measure.ExampleBasedFMeasure;
import mulan.evaluation.measure.Measure;
import net.sf.jclec.IFitness;
import net.sf.jclec.IIndividual;
import net.sf.jclec.fitness.SimpleValueFitness;
import net.sf.jclec.fitness.ValueFitnessComparator;
import net.sf.jclec.listind.MultipListGenotype;
import net.sf.jclec.listind.MultipListIndividual;
import weka.classifiers.trees.J48;

/**
 * Class implementing the evaluator for individuals (MultipListIndividuals).
 * Each individual is transformed to the corresponding multi-label classifier and then evaluated.
 * 
 * @author Jose M. Moyano
 *
 */
public class Eval extends MultipAbstractParallelEvaluator {

	/**
	 * Serialization constant
	 */
	private static final long serialVersionUID = 3488129084072488337L;

	/**
	 *  Indicates if fitness is to maximize 
	 */
	private boolean maximize = true;
	
	/**
	 *  Fitness comparator 
	 */
	private Comparator<IFitness> COMPARATOR = new ValueFitnessComparator(!maximize);	
	
	/**
	 * Datasets to build the models
	 */
	MultiLabelInstances [] trainData;
	
	/**
	 * Dataset to evaluate the model
	 */
	MultiLabelInstances evalData = null;
	
	/**
	 * Table including the fitness of all individuals
	 */
	Hashtable<String, Double> tableFitness;
	
	/**
	 * Table including the fitness of all individuals
	 */
	Hashtable<String, MultiLabelLearner> tableClassifiers;
	
	/**
	 * MultiLabelLearner
	 */
	MultiLabelLearner baseLearner;
	
	
	/**
	 * Constructor
	 */
	public Eval()
	{
		super();
	}
	

	@Override
	public Comparator<IFitness> getComparator() {
		return COMPARATOR;
	}
	
	/**
	 * Setter for mlData
	 * 
	 * @param mlData Dataset to build the models
	 */
	public void setTrainData(MultiLabelInstances [] trainData) {
		this.trainData = trainData;
	}
	
	/** 
	 * Setter for evaluation data
	 * 
	 * @param evalData Evaluation dataset
	 */
	public void setEvalData(MultiLabelInstances evalData) {
		this.evalData = evalData;
	}
	
	/**
	 * Setter for tableFitness
	 * 
	 * @param tableFitness Table storing fitness of each individual
	 */
	public void setTableFitness(Hashtable<String, Double> tableFitness) {
		this.tableFitness = tableFitness;
	}
	
	/**
	 * Setter for tableClassifiers
	 * 
	 * @param tableClassifiers Table storing all ML classifiers built
	 */
	public void setTableClassifiers(Hashtable<String, MultiLabelLearner> tableClassifiers) {
		this.tableClassifiers = tableClassifiers;
	}
	
	/**
	 * Setter for baseLearner
	 * 
	 * @param baseLearner Multi-label base learner
	 */
	public void setBaseLearner(MultiLabelLearner baseLearner) {
		this.baseLearner = baseLearner;
	}
	
	
	
	@Override
	protected void evaluate(IIndividual ind) 
	{
		//Individual genotype (list)
		MultipListGenotype gen = ((MultipListIndividual)ind).getGenotype();
		
		double fitness = -1;
		
		for(Integer g : gen.genotype) {
			fitness += g;
		}
		
		//String key for tables
		String s = gen.toString();
		
		if(tableFitness.containsKey(s)) {
			fitness = tableFitness.get(s);
		}
		else {
			try {
				int subpop = gen.subpop;
				//Get corresponding training data filtered with labels of genotype
				DatasetTransformation dt = new DatasetTransformation();
				MultiLabelInstances newData = dt.transformDataset(trainData[subpop], gen.genotype);
	
				//Build classifier
				MultiLabelLearner mll = null;
				mll = new LabelPowerset2(new J48());
				((LabelPowerset2)mll).setSeed(1);
				mll.build(newData);
				
				List<Measure> measures = new ArrayList<Measure>();
				measures.add(new ExampleBasedFMeasure());			
				MulanLP2Evaluator eval = new MulanLP2Evaluator();
				Evaluation results;
				
				//Evaluate individual with same data or with validation
				if(evalData == null) {
					results = eval.evaluate(mll, newData, measures);
				}
				else {
					MultiLabelInstances newVData =  dt.transformDataset(evalData, gen.genotype);
					results = eval.evaluate(mll, newVData, measures);
					newVData = null;
				}

				//Get fitness and fill tables
	     	  	fitness = results.getMeasures().get(0).getValue();
	     	  	tableFitness.put(s, fitness);
	     	  	tableClassifiers.put(s, mll.makeCopy());
	     	  	
	     	  	dt = null;
	     	  	newData = null;
	     	  	mll = null;
			} catch (Exception e) {
				e.printStackTrace();
				System.exit(1);
			}
		}
		
		//Set individual fitness
		ind.setFitness(new SimpleValueFitness(fitness));

	}

}
