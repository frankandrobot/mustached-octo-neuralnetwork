package com.neuralnetwork.cnn.layer;

import com.neuralnetwork.cnn.layer.builder.SamplingLayerBuilder;
import org.ejml.data.DenseMatrix64F;

public class SamplingLayer extends AbstractCnnLayer
{
    public SamplingLayer(SamplingLayerBuilder builder) throws IllegalArgumentException
    {
        //extract from the builder
        sharedNeuron = builder.getSharedNeuron();
        inputDim = builder.getInputDim();
        filter = builder.getFilter();

        int kernelDim = (int) Math.sqrt(sharedNeuron.getNumberOfWeights()-1);

        //checks
        if (kernelDim*kernelDim != sharedNeuron.getNumberOfWeights()-1)
            throw new IllegalArgumentException("Kernel must be square");
        if (inputDim < kernelDim)
            throw new IllegalArgumentException("Input can't be smaller than kernel");
        if (inputDim % kernelDim != 0)
            throw new IllegalArgumentException("Input size must be a multiple of the kernel size");

        double[] weights = sharedNeuron.getWeightsWithoutBias();
        double weight = weights[0];

        for(int i=1; i<weights.length; i++)
            if (weights[i] != weight)
                throw new IllegalArgumentException("Kernel weights must all be equal");

        //setup
        kernel = new DenseMatrix64F(kernelDim, kernelDim, true, weights);

        filter.setKernel(kernel);

        output = new DenseMatrix64F(
                inputDim/kernelDim,
                inputDim/kernelDim);
    }
}
