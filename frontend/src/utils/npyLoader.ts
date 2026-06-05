/**
 * npyLoader.ts
 * Lightweight parser for NumPy binary (.npy) format in TypeScript.
 */

export interface NpyArray {
  shape: number[];
  fortranOrder: boolean;
  descr: string;
  data: Float32Array | Uint8Array;
}

export function parseNpy(buffer: ArrayBuffer): NpyArray {
  const view = new DataView(buffer);
  
  // 1. Verify Magic Number: \x93NUMPY
  const magic = [
    view.getUint8(0),
    view.getUint8(1),
    view.getUint8(2),
    view.getUint8(3),
    view.getUint8(4),
    view.getUint8(5)
  ];
  
  const expectedMagic = [0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59]; // \x93NUMPY
  const isMagicValid = magic.every((val, i) => val === expectedMagic[i]);
  if (!isMagicValid) {
    throw new Error('Invalid magic number: not a NumPy .npy file');
  }

  // 2. Read version
  const major = view.getUint8(6);
  // const minor = view.getUint8(7); // unused

  // 3. Read header length
  let headerLength = 0;
  let headerStart = 10;
  if (major === 1) {
    headerLength = view.getUint16(8, true); // Little-endian
  } else if (major === 2) {
    headerLength = view.getUint32(8, true);
    headerStart = 12;
  } else {
    throw new Error(`Unsupported .npy major version: ${major}`);
  }

  // 4. Decode header text
  const headerBytes = new Uint8Array(buffer, headerStart, headerLength);
  const headerText = new TextDecoder('ascii').decode(headerBytes);

  // 5. Parse Python dictionary string
  // Format is like: {'descr': '<f4', 'fortran_order': False, 'shape': (2, 64, 128), }
  
  // Extract Descr
  const descrMatch = headerText.match(/'descr':\s*'([^']*)'/);
  if (!descrMatch) {
    throw new Error('Failed to parse descr from npy header');
  }
  const descr = descrMatch[1];

  // Extract Fortran Order
  const fortranMatch = headerText.match(/'fortran_order':\s*(True|False)/i);
  if (!fortranMatch) {
    throw new Error('Failed to parse fortran_order from npy header');
  }
  const fortranOrder = fortranMatch[1].toLowerCase() === 'true';

  // Extract Shape
  const shapeMatch = headerText.match(/'shape':\s*\(([^)]*)\)/);
  if (!shapeMatch) {
    throw new Error('Failed to parse shape from npy header');
  }
  const shapeStr = shapeMatch[1].trim();
  // Handle empty shape (scalar) or shapes with single dimensions like (128,)
  const shape = shapeStr ? shapeStr.split(',').map(s => s.trim()).filter(s => s.length > 0).map(s => parseInt(s, 10)) : [];

  // 6. Map raw binary data to TypedArray
  const dataStart = headerStart + headerLength;
  const dataBytes = buffer.byteLength - dataStart;
  
  let data: Float32Array | Uint8Array;
  
  if (descr === '<f4' || descr === '>f4') {
    const floatCount = dataBytes / 4;
    // Align ArrayBuffer offset for Float32Array (must be multiple of 4)
    // If offset is not aligned, copy slice of buffer
    if (dataStart % 4 === 0) {
      data = new Float32Array(buffer, dataStart, floatCount);
    } else {
      const slice = buffer.slice(dataStart, dataStart + floatCount * 4);
      data = new Float32Array(slice);
    }
  } else if (descr === '|b1' || descr === '|u1' || descr === '<u1') {
    data = new Uint8Array(buffer, dataStart, dataBytes);
  } else {
    throw new Error(`Unsupported npy data type descriptor: ${descr}`);
  }

  return {
    shape,
    fortranOrder,
    descr,
    data
  };
}

export async function fetchNpy(url: string): Promise<NpyArray> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch npy file: ${response.statusText}`);
  }
  const buffer = await response.arrayBuffer();
  return parseNpy(buffer);
}
