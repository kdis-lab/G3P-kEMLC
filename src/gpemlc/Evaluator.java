package gpemlc;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.Hashtable;
import java.util.List;

import mulan.classifier.MultiLabelLearner;
import mulan.classifier.transformation.LabelPowerset2;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluation;
import mulan.evaluation.MulanLP2Evaluator;
import mulan.evaluation.measure.ExampleBasedFMeasure;
import mulan.evaluation.measure.Measure;
import net.sf.jclec.IFitness;
import net.sf.jclec.IIndividual;
import net.sf.jclec.base.AbstractParallelEvaluator;
import net.sf.jclec.fitness.SimpleValueFitness;
import net.sf.jclec.fitness.ValueFitnessComparator;
import net.sf.jclec.stringtree.StringTreeIndividual;
import weka.classifiers.trees.J48;

/**
 * Class implementing the evaluator for individuals (MultipListIndividuals).
 * Each individual is transformed to the corresponding multi-label classifier and then evaluated.
 * 
 * @author Jose M. Moyano
 *
 */
public class Evaluator extends AbstractParallelEvaluator {

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
	 * Constructor
	 */
	public Evaluator()
	{
		super();
	}
	

	@Override
	public Comparator<IFitness> getComparator() {
		return COMPARATOR;
	}
	
	
	@Override
	protected void evaluate(IIndividual ind) 
	{
		//Individual genotype (list)
		String gen = ((StringTreeIndividual)ind).getGenotype();
		
		Utils utils = new Utils();
		double fitness = utils.countLeaves(gen);
		
		//Set individual fitness
		ind.setFitness(new SimpleValueFitness(fitness));

	}

}
