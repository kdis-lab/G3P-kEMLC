package coeaglet.algorithm;

import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import net.sf.jclec.IIndividual;
import net.sf.jclec.base.AbstractEvaluator;

/**
 * Parallel abstract evaluator for multi-population algorithms
 * 
 * @author Alberto Cano
 * @author Sebastian Ventura
 * @author Jose M. Moyano
 */

public abstract class MultipAbstractParallelEvaluator extends AbstractEvaluator
{
	/////////////////////////////////////////////////////////////////
	// ------------------------------------------------- Constructors
	/////////////////////////////////////////////////////////////////
	
	private static final long serialVersionUID = 3270366292953409655L;

	public long executionTime = 0;
	
	/**
	 * Empty constructor.
	 */
	
	public MultipAbstractParallelEvaluator() 
	{
		super();
	}

	/////////////////////////////////////////////////////////////////
	// ----------------------------------------------- Public methods
	/////////////////////////////////////////////////////////////////

	/**
	 * For all individuals in "inds" array: if individual fitness  is
	 * null, then evaluate this individual.
	 * 
	 * This method is final. Is anyone wants implement this method in
	 * another way should create a new IEvaluator class.
	 * 
	 * {@inheritDoc}
	 */
	
	public void evaluate(List<IIndividual> inds)
	{
		long time = System.currentTimeMillis();
		
		ExecutorService threadExecutor = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
		
		for (IIndividual ind : inds)
		{
			if (ind.getFitness() == null)
			{
				threadExecutor.execute(new evaluationThread(ind));
				numberOfEvaluations++;
			}
		}
		
		threadExecutor.shutdown();
		
		try
		{
			if (!threadExecutor.awaitTermination(30, TimeUnit.DAYS))
				System.out.println("Threadpool timeout occurred");
		}
		catch (InterruptedException ie)
		{
			System.out.println("Threadpool prematurely terminated due to interruption in thread that created pool");
		}

		executionTime += System.currentTimeMillis() - time;
	}
	
	public void evaluateMultip(List<List<IIndividual>> bset)
	{
		long time = System.currentTimeMillis();
		
		ExecutorService threadExecutor = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
		
		for(List<IIndividual> subpop : bset) {
			for (IIndividual ind : subpop)
			{
				if (ind.getFitness() == null)
				{
					threadExecutor.execute(new evaluationThread(ind));
					numberOfEvaluations++;
				}
			}
		}
		
		threadExecutor.shutdown();
		
		try
		{
			if (!threadExecutor.awaitTermination(30, TimeUnit.DAYS))
				System.out.println("Threadpool timeout occurred");
		}
		catch (InterruptedException ie)
		{
			System.out.println("Threadpool prematurely terminated due to interruption in thread that created pool");
		}

		executionTime += System.currentTimeMillis() - time;
	}
	
	/////////////////////////////////////////////////////////////////
	// -------------------------------------------- Evaluation Thread
	/////////////////////////////////////////////////////////////////
	
	private class evaluationThread extends Thread
	{
		private IIndividual ind;
		
	    public evaluationThread(IIndividual ind)
	    {
	        this.ind = ind;
	    }
	    
	    public void run()
	    {
	    	evaluate(ind);
	    }
    }
}