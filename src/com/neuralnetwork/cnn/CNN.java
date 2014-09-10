package com.neuralnetwork.cnn;

import com.neuralnetwork.core.interfaces.INeuralNetwork;
import org.ejml.data.DenseMatrix64F;

public class CNN implements INeuralNetwork<DenseMatrix64F>
{
    protected CNNConnections[] connections;

    @Override
    public DenseMatrix64F generateOutput(DenseMatrix64F input) {

        for(int i=0; i<connections.length; i++)
        {
            CNNConnections conns = connections[i];

        }
    }

    @Override
    public DenseMatrix64F generateYoutput(DenseMatrix64F denseMatrix64F) {
        return null;
    }
}
