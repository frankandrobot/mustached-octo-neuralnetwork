package com.neuralnetwork.cnn.singlelayer;

import com.neuralnetwork.cnn.filter.IFilter;
import com.neuralnetwork.core.Dimension;
import com.neuralnetwork.core.interfaces.IActivationFunction;
import com.neuralnetwork.core.interfaces.ICnnMap;
import com.neuralnetwork.core.neuron.Neuron;
import org.ejml.data.DenseMatrix64F;
import org.ejml.ops.CommonOps;

import java.util.List;

/**
 * A {@link com.neuralnetwork.cnn.CNNLayer} is made up of many {@link AbstractCNNSingleLayer}s
  */
public abstract class AbstractCNNSingleLayer<Builder extends AbstractCNNSingleLayerBuilder> implements ICnnMap
{
    /**
     * the kernel is made of weights from this neuron.
     * the shared neuron contains most of the info necessary to build the layer
     */
    protected Neuron[] aSharedNeurons;
    protected DenseMatrix64F[] aKernels;

    /**
     * one dimension of the input. assumes input is square
     */
    protected int inputDim;
    protected int numberOfInputs;

    /**
     * matrix used to store latest output
     */
    protected DenseMatrix64F output;
    protected DenseMatrix64F tmp;

    /**
     * the filter used to convolve
     */
    protected IFilter[] aFilters;


    public AbstractCNNSingleLayer(Builder builder)
    {
        //extract from builder
        aSharedNeurons = builder.aSharedNeurons;
        inputDim = builder.inputDim;
        numberOfInputs = builder.numberOfInputs;
        aKernels = builder.aKernels;

        createFilters(builder);
        createOutputMatrix(builder);

        tmp = new DenseMatrix64F(output.numRows, output.numCols);
    }


    abstract protected void createFilters(Builder builder);

    abstract protected void createOutputMatrix(Builder builder);


    DenseMatrix64F generateOutput(DenseMatrix64F input)
    {
        return generateOutput(new DenseMatrix64F[] { input });
    }

    @Override
    public DenseMatrix64F generateOutput(DenseMatrix64F[] input)
    {
        output = generateInducedLocalField(input);

        //apply activation function to output
        IActivationFunction phi = aSharedNeurons[0].phi();

        double[] _output = output.getData();

        for(int i=0; i<_output.length; i++)
        {
            _output[i] = phi.apply(_output[i]);
        }

        return output;
    }

    @Override
    public DenseMatrix64F generateInducedLocalField(DenseMatrix64F[] input)
    {
        //first the weights
        CommonOps.fill(output, 0);

        for(int i=0; i<input.length; i++)
        {
            aFilters[i].convolve(input[i], tmp);
            CommonOps.addEquals(output, tmp);
        }


        //now the biases
        double bias = aSharedNeurons[0].getBias();

        double[] o = output.getData();

        for(int i=0; i<o.length; i++)
        {
            o[i] += bias;
        }

        return output;
    }

    @Override
    public void validateInputs(List<ICnnMap> inputMaps)
    {
        Dimension inputDims = getInputDim();

        int len =0;

        for(ICnnMap inputMap:inputMaps)
        {
            Dimension outputDims = inputMap.getOutputDim();

            if (!inputDims.equals(outputDims))
                throw new IllegalArgumentException(
                        "Input dimensions must match. Failure in ´input map "+len);

            len++;
        }
    }

    /**
     * @deprecated
     *
     * @return
     */
    public DenseMatrix64F getOutput()
    {
        return output;
    }

    @Override
    public DenseMatrix64F getWeightMatrix()
    {
        return null;
    }

    @Override
    public IActivationFunction.IDifferentiableFunction getImpulseFunction() {
        return null;
    }

    public Dimension getInputDim()
    {
        return new Dimension(inputDim, inputDim);
    }

    public Dimension getOutputDim()
    {
        return new Dimension(output.numRows, output.numCols);
    }

    public int getNumberOfInputs()
    {
        return numberOfInputs;
    }
}
