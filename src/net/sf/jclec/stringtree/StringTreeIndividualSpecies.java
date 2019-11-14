package net.sf.jclec.stringtree;

import net.sf.jclec.IConfigure;
import net.sf.jclec.binarray.BinArrayIndividual;
import net.sf.jclec.binarray.BinArrayIndividualSpecies;
import net.sf.jclec.binarray.BinArraySpecies;

import java.util.StringTokenizer;

import org.apache.commons.lang.builder.EqualsBuilder;

import org.apache.commons.configuration.Configuration;

/**
 * BinArrayIndividual species.
 * 
 * Species for individuals of type "BinArrayIndividual". This species set its 
 * schema explicitly (using the setSchema() method). 
 * 
 * @author Sebastian Ventura 
 */

public final class StringTreeIndividualSpecies extends StringTreeSpecies implements IConfigure 
{
	/////////////////////////////////////////////////////////////////
	// --------------------------------------- Serialization constant
	/////////////////////////////////////////////////////////////////
	
	/** Generated by Eclipse */
	
	private static final long serialVersionUID = 1849608890559693424L;

	/////////////////////////////////////////////////////////////////
	// ------------------------------------------------- Constructors
	/////////////////////////////////////////////////////////////////
	
	/**
	 * Empty constructor
	 */
	
	public StringTreeIndividualSpecies() 
	{
		super();
	}

	/**
	 * Constructor that sets individual species
	 */
	
	public StringTreeIndividualSpecies(String genotypeSchema) 
	{
		super();
		setGenotypeSchema(genotypeSchema);
	}
	
	/////////////////////////////////////////////////////////////////
	// ----------------------------------------------- Public methods
	/////////////////////////////////////////////////////////////////
	
	// Setting properties
	
	/**
	 * Sets genotype schema.
	 * 
	 * This method is used  in marshall/unmarshall  and configuration
	 * methods. 
	 * 
	 * @param genotypeSchema New genotype schema.
	 */
	
	public void setGenotypeSchema(String genotypeSchema)
	{
		this.genotypeSchema = genotypeSchema;
	}
	
	// Factory method
	
	/**
	 * Factory method.
	 * 
	 * @param genotype Individual genotype
	 */
	
	public StringTreeIndividual createIndividual(String genotype) 
	{
		return new StringTreeIndividual(genotype);
	}

	// IConfigure interface

	/**
	 * Configuration parameters for this species are:
	 * 
	 * <ul>
	 * <li>
	 * <code>[@genotype-length] (int)</code></p>
	 *  Genotype length. If this parameter is set, 'genotype-schema'
	 *  parameter is ignored and all schema positions are set to '*'.
	 * </li>
	 * <li>
	 * <code>genotype-schema  (String)</code></p>
	 * Genotype schema. This parameter contains characters '1', '0' 
	 * and '*' to represent schema elements...   
	 * </li>
	 * </ul> 
	 */
	
	public void configure(Configuration configuration) 
	{
		// Genotype schema
//		byte [] genotypeSchema;
//		// Get 'length' parameter
//		int genotypeLength = configuration.getInt("[@genotype-length]", 0);
//		if (genotypeLength == 0) {
//			// Genotype schema string
//			String genotypeSchemaString = configuration.getString("genotype-schema");
//			// Parses genotype-schema
//			StringTokenizer st = new StringTokenizer(genotypeSchemaString);
//			int gl = st.countTokens();
//			genotypeSchema = new byte[gl];
//			for (int i=0; i<gl; i++) {
//				String nt = st.nextToken(); 
//				if (nt.equals("*")) {
//					genotypeSchema[i] = -1;
//				}
//				else {
//					genotypeSchema[i] = Byte.parseByte(nt);
//				}
//			}			
//		}
//		else {
//			// Allocate space for schema
//			genotypeSchema = new byte[genotypeLength]; 
//			// Set default values for schema
//			for (int i=0; i<genotypeLength; i++) {
//				genotypeSchema[i] = -1;
//			}
//		}
		// Set genotype schema
		setGenotypeSchema(new String());
	}

	/////////////////////////////////////////////////////////////////
	// ----------------------------------- Overwriting Object methods
	/////////////////////////////////////////////////////////////////
	
	/**
	 * {@inheritDoc}
	 */
	
	public boolean equals(Object other)
	{
		if (other instanceof StringTreeIndividualSpecies) {
			EqualsBuilder eb = new EqualsBuilder();
			StringTreeIndividualSpecies baoth = (StringTreeIndividualSpecies) other;
			eb.append(this.genotypeSchema, baoth.genotypeSchema);
			return eb.isEquals();
		}
		else {
			return false;
		}
	}
}

