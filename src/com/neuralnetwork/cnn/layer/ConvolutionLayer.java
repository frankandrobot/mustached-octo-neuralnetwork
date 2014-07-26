package com.neuralnetwork.cnn.layer;

import com.neuralnetwork.cnn.layer.builder.ConvolutionLayerBuilder;
import com.neuralnetwork.cnn.layer.builder.FeatureMapBuilder;
import org.ejml.data.DenseMatrix64F;

public class ConvolutionLayer extends BaseCnnLayer
{
    public ConvolutionLayer(ConvolutionLayerBuilder builder) throws IllegalArgumentException
    {
        //extract from builder
        sharedNeuron = builder.getSharedNeuron();
        inputDim = builder.getInputDim();
        filter = builder.getFilter();

        int kernelDim = (int) Math.sqrt(sharedNeuron.getNumberOfWeights()-1);

        //checks
        if (kernelDim*kernelDim != sharedNeuron.getNumberOfWeights()-1)
            throw new IllegalArgumentException("Kernel must be square");
        if (inputDim < kernelDim)
            throw new IllegalArgumentException("Input can't be smaller than kernel");

        //build
        double[] weights = sharedNeuron.getWeightsWithoutBias().getData();
        kernel = new DenseMatrix64F(kernelDim, kernelDim, true, weights);

        filter.setKernel(kernel);

        output = new DenseMatrix64F(
                inputDim-kernelDim+1,
                inputDim-kernelDim+1);
    }
}
