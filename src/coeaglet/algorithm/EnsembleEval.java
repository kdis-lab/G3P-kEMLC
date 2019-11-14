package coeaglet.algorithm;

import java.util.ArrayList;
import java.util.List;

import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluation;
import mulan.evaluation.MulanEnsembleEvaluator;
import mulan.evaluation.measure.ExampleBasedFMeasure;
import mulan.evaluation.measure.Measure;

/**
 * Class implementing the evaluator of the Ensemble
 * 
 * @author Jose M. Moyano
 *
 */
public class EnsembleEval {

	/**
	 * Dataset to build and evaluate the model
	 */
	MultiLabelInstances mlData;
	
	/**
	 * Ensemble to evaluate
	 */
	Ensemble ensemble;
	
	/**
	 * Constructor
	 */
	public EnsembleEval(Ensemble ensemble, MultiLabelInstances mlData)
	{
		super();
		this.ensemble = ensemble;
		this.mlData = mlData;
	}
	
	/**
	 * Setter for mlData
	 * 
	 * @param mlData Dataset to build and evaluate model
	 */
	public void setMlData(MultiLabelInstances mlData) {
		this.mlData = mlData;
	}

	/**
	 * Evaluate the ensemble. Using ExF
	 * 
	 * @return Fitness of the ensemble. 
	 */
	protected double evaluate() 
	{
		ensemble.resetSeed();
		
		double fitness = -1;
		
		try {
			List<Measure> measures = new ArrayList<Measure>();
			measures.add(new ExampleBasedFMeasure());
					
			MulanEnsembleEvaluator eval = new MulanEnsembleEvaluator();
			Evaluation results;
			
			results = eval.evaluate(ensemble, mlData, measures);
			fitness = results.getMeasures().get(0).getValue();
			
			ensemble.setFitness(fitness);
			
		} catch (Exception e) {
			e.printStackTrace();
		}

		return fitness;
	}

}
