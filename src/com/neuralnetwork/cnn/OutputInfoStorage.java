package com.neuralnetwork.cnn;

import com.neuralnetwork.core.interfaces.IMatrixNeuralLayer;

import java.util.HashMap;

class OutputInfoStorage
{
    private HashMap<IMatrixNeuralLayer,HashMap<IMatrixNeuralLayer,OutputInfo>> storage
            = new HashMap<IMatrixNeuralLayer, HashMap<IMatrixNeuralLayer, OutputInfo>>();

    public void set(IMatrixNeuralLayer a, IMatrixNeuralLayer b, OutputInfo outputInfo)
    {
        HashMap<IMatrixNeuralLayer,OutputInfo> row = storage.get(a);

        if (row == null)
            storage.put(a, new HashMap<IMatrixNeuralLayer,OutputInfo>());

        row.put(b, outputInfo);
    }

    public OutputInfo get(IMatrixNeuralLayer a, IMatrixNeuralLayer b)
    {
        HashMap<IMatrixNeuralLayer,OutputInfo> row = storage.get(a);

        if (row != null)
            return row.get(b);

        return null;
    }
}
