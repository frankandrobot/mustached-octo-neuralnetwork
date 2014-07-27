package com.neuralnetwork.cnn;

import com.neuralnetwork.core.interfaces.IActivationFunction;
import com.neuralnetwork.core.interfaces.INeuralNetwork;
import org.ejml.data.DenseMatrix64F;

public class Cnn
{
    private final double momemtumParam;
    private final double learningParam;
    private final INeuralNetwork.IMatrixNeuralNetwork[] aLayers;
    private final IActivationFunction globalPhi;

    private DenseMatrix64F output;

    public Cnn(CnnBuilder netBuilder)
    {
        this.momemtumParam = netBuilder.getMomentumParam();
        this.learningParam = netBuilder.getLearningParam();
        this.aLayers = netBuilder.getLayers();
        this.output = aLayers[aLayers.length-1].getOutput();

        //check layers output---inputs must match
        for(int i=0; i<aLayers.length-1; i++)
        {
            INeuralNetwork.IMatrixNeuralNetwork layer = aLayers[i];
            INeuralNetwork.IMatrixNeuralNetwork layerNext = aLayers[i+1];
            if (layer.getOutput().numRows != layerNext.getInputDim())
                throw new IllegalArgumentException("Output/inputs in layers must match: layers "+i+","+i+1);
        }
    }

    public DenseMatrix64F generateOutput(DenseMatrix64F input)
    {
        DenseMatrix64F _input = input;
        DenseMatrix64F _output = null;

        int len = -1;
        while (++len < aLayers.length)
        {
            _output = aLayers[len].generateOutput(_input);
            _input = _output;
        }
        return _output;
    }
}
