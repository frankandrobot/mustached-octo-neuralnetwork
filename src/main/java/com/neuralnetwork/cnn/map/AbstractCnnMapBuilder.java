package com.neuralnetwork.cnn.map;

import com.neuralnetwork.cnn.filter.IFilter;
import com.neuralnetwork.core.interfaces.IActivationFunction;
import com.neuralnetwork.core.neuron.Neuron;
import org.ejml.data.DenseMatrix64F;

abstract class AbstractCnnMapBuilder<Builder extends AbstractCnnMapBuilder,Map extends AbstractCnnMap>
{
    int inputDim;

    int numberOfInputs;

    Neuron[] aSharedNeurons;

    IFilter[] aFilters;

    DenseMatrix64F[] aKernels;

    int kernelDim;


    /**
     * Assumes a square input
     * @param inputSize
     * @return
     */
    public Builder set1DInputSize(int inputSize)
    {
        this.inputDim = inputSize;

        return (Builder) this;
    }

    public Builder setNeuron(Neuron... neuron)
    {
        this.aSharedNeurons = neuron;

        return (Builder) this;
    }


    public Map build()
    {
        validate();

        aKernels = new DenseMatrix64F[aFilters.length];
        kernelDim = (int) Math.sqrt(aSharedNeurons[0].getNumberOfWeights()-1);
        numberOfInputs = aFilters.length;

        for(int i=0; i<numberOfInputs; i++)
        {
            double[] weights = aSharedNeurons[i].getWeightsWithoutBias();
            aKernels[i] = new DenseMatrix64F(kernelDim, kernelDim, true, weights);

            aFilters[i].setKernel(aKernels[i]);
        }

        return buildMap();
    }


    private void validate()
    {
        if (aSharedNeurons.length != aFilters.length)
            throw new IllegalArgumentException("Need as many neurons as filters");


        double bias = aSharedNeurons[0].getBias();
        double numberOfWeights = aSharedNeurons[0].getNumberOfWeights();
        IActivationFunction.IDifferentiableFunction phi = aSharedNeurons[0].phi();

        for(Neuron neuron:aSharedNeurons)
        {
            if (neuron.getBias() != bias)
                throw new IllegalArgumentException("The biases must all be the same");

            if (neuron.getNumberOfWeights() != numberOfWeights)
                throw new IllegalArgumentException("The number of weights must all be the same");

            if (neuron.phi() != phi)
                throw new IllegalArgumentException("The activation functions must all be the same");
        }


        for(Neuron neuron:aSharedNeurons)
        {
            int kernelDim = (int) Math.sqrt(neuron.getNumberOfWeights() - 1);

            if (kernelDim * kernelDim != neuron.getNumberOfWeights() - 1)
                throw new IllegalArgumentException("Kernel must be square");

            if (inputDim < kernelDim)
                throw new IllegalArgumentException("Input can't be smaller than kernel");
        }
    }

    abstract protected Map buildMap();

}
