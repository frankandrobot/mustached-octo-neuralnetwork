package com.neuralnetwork.cnn;

import com.neuralnetwork.core.interfaces.IMatrixNeuralLayer;
import org.ejml.data.DenseMatrix64F;
import org.ejml.ops.CommonOps;

import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

class MapInfo
{
    /**
     * could be null
     */
    IMatrixNeuralLayer map;

    List<MapInfo> lInputMaps = new LinkedList<MapInfo>();

    DenseMatrix64F yOutput;


    public MapInfo(IMatrixNeuralLayer map)
    {
        if (map != null)
        {
            this.map = map;
            yOutput = new DenseMatrix64F(map.getOutputDim().rows, map.getOutputDim().cols);
        }
    }

    public MapInfo(DenseMatrix64F input)
    {
        yOutput = input;
    }

    /**
     * Valid only if has map
     */
    public void generateOutput()
    {
        for(MapInfo inputMap:lInputMaps)
        {
            DenseMatrix64F _yOutput = inputMap.yOutput;

            DenseMatrix64F output = map.generateOutput(_yOutput);

            CommonOps.addEquals(this.yOutput, output);
        }
    }
}
