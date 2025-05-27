function getProjectionMatrix(fx, fy, width, height) {
    const znear = 0.01;
    const zfar = 100;
    return [
        [(2 * fx) / width, 0, 0, 0],
        [0, -(2 * fy) / height, 0, 0],
        [0, 0, zfar / (zfar - znear), 1],
        [0, 0, -(zfar * znear) / (zfar - znear), 0],
    ].flat();
}

function multiply4(a, b) {
    return [
        b[0] * a[0] + b[1] * a[4] + b[2] * a[8] + b[3] * a[12],
        b[0] * a[1] + b[1] * a[5] + b[2] * a[9] + b[3] * a[13],
        b[0] * a[2] + b[1] * a[6] + b[2] * a[10] + b[3] * a[14],
        b[0] * a[3] + b[1] * a[7] + b[2] * a[11] + b[3] * a[15],
        b[4] * a[0] + b[5] * a[4] + b[6] * a[8] + b[7] * a[12],
        b[4] * a[1] + b[5] * a[5] + b[6] * a[9] + b[7] * a[13],
        b[4] * a[2] + b[5] * a[6] + b[6] * a[10] + b[7] * a[14],
        b[4] * a[3] + b[5] * a[7] + b[6] * a[11] + b[7] * a[15],
        b[8] * a[0] + b[9] * a[4] + b[10] * a[8] + b[11] * a[12],
        b[8] * a[1] + b[9] * a[5] + b[10] * a[9] + b[11] * a[13],
        b[8] * a[2] + b[9] * a[6] + b[10] * a[10] + b[11] * a[14],
        b[8] * a[3] + b[9] * a[7] + b[10] * a[11] + b[11] * a[15],
        b[12] * a[0] + b[13] * a[4] + b[14] * a[8] + b[15] * a[12],
        b[12] * a[1] + b[13] * a[5] + b[14] * a[9] + b[15] * a[13],
        b[12] * a[2] + b[13] * a[6] + b[14] * a[10] + b[15] * a[14],
        b[12] * a[3] + b[13] * a[7] + b[14] * a[11] + b[15] * a[15],
    ];
}

function invert4(a) {
    let b00 = a[0] * a[5] - a[1] * a[4];
    let b01 = a[0] * a[6] - a[2] * a[4];
    let b02 = a[0] * a[7] - a[3] * a[4];
    let b03 = a[1] * a[6] - a[2] * a[5];
    let b04 = a[1] * a[7] - a[3] * a[5];
    let b05 = a[2] * a[7] - a[3] * a[6];
    let b06 = a[8] * a[13] - a[9] * a[12];
    let b07 = a[8] * a[14] - a[10] * a[12];
    let b08 = a[8] * a[15] - a[11] * a[12];
    let b09 = a[9] * a[14] - a[10] * a[13];
    let b10 = a[9] * a[15] - a[11] * a[13];
    let b11 = a[10] * a[15] - a[11] * a[14];
    let det =
        b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
    if (!det) return null;
    return [
        (a[5] * b11 - a[6] * b10 + a[7] * b09) / det,
        (a[2] * b10 - a[1] * b11 - a[3] * b09) / det,
        (a[13] * b05 - a[14] * b04 + a[15] * b03) / det,
        (a[10] * b04 - a[9] * b05 - a[11] * b03) / det,
        (a[6] * b08 - a[4] * b11 - a[7] * b07) / det,
        (a[0] * b11 - a[2] * b08 + a[3] * b07) / det,
        (a[14] * b02 - a[12] * b05 - a[15] * b01) / det,
        (a[8] * b05 - a[10] * b02 + a[11] * b01) / det,
        (a[4] * b10 - a[5] * b08 + a[7] * b06) / det,
        (a[1] * b08 - a[0] * b10 - a[3] * b06) / det,
        (a[12] * b04 - a[13] * b02 + a[15] * b00) / det,
        (a[9] * b02 - a[8] * b04 - a[11] * b00) / det,
        (a[5] * b07 - a[4] * b09 - a[6] * b06) / det,
        (a[0] * b09 - a[1] * b07 + a[2] * b06) / det,
        (a[13] * b01 - a[12] * b03 - a[14] * b00) / det,
        (a[8] * b03 - a[9] * b01 + a[10] * b00) / det,
    ];
}

function rotate4(a, rad, x, y, z) {
    let len = Math.hypot(x, y, z);
    x /= len;
    y /= len;
    z /= len;
    let s = Math.sin(rad);
    let c = Math.cos(rad);
    let t = 1 - c;
    let b00 = x * x * t + c;
    let b01 = y * x * t + z * s;
    let b02 = z * x * t - y * s;
    let b10 = x * y * t - z * s;
    let b11 = y * y * t + c;
    let b12 = z * y * t + x * s;
    let b20 = x * z * t + y * s;
    let b21 = y * z * t - x * s;
    let b22 = z * z * t + c;
    return [
        a[0] * b00 + a[4] * b01 + a[8] * b02,
        a[1] * b00 + a[5] * b01 + a[9] * b02,
        a[2] * b00 + a[6] * b01 + a[10] * b02,
        a[3] * b00 + a[7] * b01 + a[11] * b02,
        a[0] * b10 + a[4] * b11 + a[8] * b12,
        a[1] * b10 + a[5] * b11 + a[9] * b12,
        a[2] * b10 + a[6] * b11 + a[10] * b12,
        a[3] * b10 + a[7] * b11 + a[11] * b12,
        a[0] * b20 + a[4] * b21 + a[8] * b22,
        a[1] * b20 + a[5] * b21 + a[9] * b22,
        a[2] * b20 + a[6] * b21 + a[10] * b22,
        a[3] * b20 + a[7] * b21 + a[11] * b22,
        ...a.slice(12, 16),
    ];
}

function translate4(a, x, y, z) {
    return [
        ...a.slice(0, 12),
        a[0] * x + a[4] * y + a[8] * z + a[12],
        a[1] * x + a[5] * y + a[9] * z + a[13],
        a[2] * x + a[6] * y + a[10] * z + a[14],
        a[3] * x + a[7] * y + a[11] * z + a[15],
    ];
}

function createWorker(self) {
  // We’ll read a binary_little_endian PLY with:
  //   element vertex N; property float x,y,z
  //   element face  M; property list uchar int vertex_indices
  //                    property uchar red,green,blue
  //                    property float quality ← alpha
  let buffer, vertexCount=0, faceCount=0;
  let positions,     // Float32Array [x0,y0,z0, x1,y1,z1, …]
      faceIndices,   // Uint32Array [i0,i1,i2, …]
      faceColors;    // Float32Array [r0,g0,b0,a0, …]
  let depthIndex = new Uint32Array();

  function parsePLY(input) {
    const dv = new DataView(input);
    // 1) read header text
    const header = new TextDecoder().decode(new Uint8Array(input,0,64*1024));
    const eoh = header.indexOf('end_header') + 11;
    const lines = header.slice(0,eoh).split(/\r?\n/);
    for (let l of lines) {
      let m;
      if (m = l.match(/^element vertex (\d+)/)) vertexCount = +m[1];
      if (m = l.match(/^element face (\d+)/))   faceCount   = +m[1];
    }
    // 2) allocate
    positions   = new Float32Array(vertexCount*3);
    faceIndices = new Uint32Array(faceCount*3);
    faceColors  = new Float32Array(faceCount*4);

    // 3) scan binary body
    let off = eoh;
    // vertices
    for (let v=0; v<vertexCount; v++) {
      positions[3*v+0] = dv.getFloat32(off, true); off+=4;
      positions[3*v+1] = dv.getFloat32(off, true); off+=4;
      positions[3*v+2] = dv.getFloat32(off, true); off+=4;
    }
    // faces
    for (let f=0; f<faceCount; f++) {
      const n = dv.getUint8(off); off++;
      if (n !== 3) {
        off += n*4 + 3 + 4;
        faceIndices.set([0,0,0], f*3);
        faceColors .set([0,0,0,0], f*4);
        continue;
      }
      const i0 = dv.getInt32(off,true); off+=4;
      const i1 = dv.getInt32(off,true); off+=4;
      const i2 = dv.getInt32(off,true); off+=4;
      faceIndices.set([i0,i1,i2], f*3);

      const r = dv.getUint8(off++)/255;
      const g = dv.getUint8(off++)/255;
      const b = dv.getUint8(off++)/255;
      const a = dv.getFloat32(off,true); off+=4;
      faceColors.set([r,g,b,a], f*4);
    }
  }

  // 16-bit counting sort on centroids
  function sortFaces(viewProj) {
    if (!positions) return;
    const N = faceCount;
    // compute depths
    const depths = new Float32Array(N);
    let mi= Infinity, ma=-Infinity;
    for (let f=0; f<N; f++) {
      const i0 = faceIndices[3*f+0],
            i1 = faceIndices[3*f+1],
            i2 = faceIndices[3*f+2];
      // centroid
      const cx = (positions[3*i0+0]+positions[3*i1+0]+positions[3*i2+0])/3;
      const cy = (positions[3*i0+1]+positions[3*i1+1]+positions[3*i2+1])/3;
      const cz = (positions[3*i0+2]+positions[3*i1+2]+positions[3*i2+2])/3;
      // project
      const w0 = viewProj[ 0]*cx + viewProj[ 4]*cy + viewProj[ 8]*cz + viewProj[12];
      const w1 = viewProj[ 1]*cx + viewProj[ 5]*cy + viewProj[ 9]*cz + viewProj[13];
      const w2 = viewProj[ 2]*cx + viewProj[ 6]*cy + viewProj[10]*cz + viewProj[14];
      const w3 = viewProj[ 3]*cx + viewProj[ 7]*cy + viewProj[11]*cz + viewProj[15];
      const d = (w2/w3);
      depths[f]=d;
      if (d<mi) mi=d;
      if (d>ma) ma=d;
    }
    // quantize into 0…65535
    const inv = 0xFFFF/(ma-mi||1);
    const counts  = new Uint32Array(1<<16);
    const offsets = new Uint32Array(1<<16);
    depthIndex    = new Uint32Array(N);
    for (let f=0; f<N; f++) {
      const q = ((depths[f]-mi)*inv)|0;
      counts[q]++;
      depthIndex[f]=q;
    }
    // prefix sum
    for (let i=1; i<counts.length; i++) offsets[i]=offsets[i-1]+counts[i-1];
    // scatter
    const sorted = new Uint32Array(N);
    for (let f=0; f<N; f++) {
      const b = depthIndex[f];
      sorted[offsets[b]++] = f;
    }
    // reverse for back→front
    return sorted.reverse();
  }

  // worker.onmessage
  let sortedIDs;
  self.onmessage = e => {
    if (e.data.ply) {
      parsePLY(e.data.ply);
        // send back all the parsed buffers plus faceCount:
        self.postMessage({
            init: true,
            positions: positions.buffer,
            faceIndices: faceIndices.buffer,
            faceColors: faceColors.buffer,
            faceCount
        }, [
            positions.buffer,
            faceColors.buffer
        ]);
    }
    else if (e.data.viewProj) {
      const order = sortFaces(e.data.viewProj);
      self.postMessage({ depthOrder: order.buffer }, [order.buffer]);
    }
  };
}
// ─────────────── vertexShaderSource ───────────────
const vertexShaderSource = `#version 300 es
precision highp float;
precision highp int;

uniform mat4 projection;
uniform mat4 view;

in vec3 a_position;   // the 3D position of each vertex
in vec4 a_color;      // per-vertex color+alpha

out vec4 vColor;

void main() {
  // transform into clip‐space
  gl_Position = projection * view * vec4(a_position, 1.0);
  // pass RGBA through to the fragment shader
  vColor = a_color;
}
`;

// ─────────────── fragmentShaderSource ───────────────
const fragmentShaderSource = `#version 300 es
precision highp float;

in vec4 vColor;
out vec4 outColor;

void main() {
  // just output the interpolated color+alpha
  outColor = vColor;
}
`;

let defaultViewMatrix = [
    0.47, 0.04, 0.88, 0, -0.11, 0.99, 0.02, 0, -0.88, -0.11, 0.47, 0, 0.07,
    0.03, 6.55, 1,
];
let viewMatrix = defaultViewMatrix;
async function main() {
    const canvas = document.getElementById("canvas");
    const fps = document.getElementById("fps");

    const gl = canvas.getContext("webgl2", {
        antialias: false,
    });

    const vertexShader = gl.createShader(gl.VERTEX_SHADER);
    gl.shaderSource(vertexShader, vertexShaderSource);
    gl.compileShader(vertexShader);
    if (!gl.getShaderParameter(vertexShader, gl.COMPILE_STATUS))
        console.error(gl.getShaderInfoLog(vertexShader));

    const fragmentShader = gl.createShader(gl.FRAGMENT_SHADER);
    gl.shaderSource(fragmentShader, fragmentShaderSource);
    gl.compileShader(fragmentShader);
    if (!gl.getShaderParameter(fragmentShader, gl.COMPILE_STATUS))
        console.error(gl.getShaderInfoLog(fragmentShader));

    const program = gl.createProgram();
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);
    gl.useProgram(program);

    if (!gl.getProgramParameter(program, gl.LINK_STATUS))
        console.error(gl.getProgramInfoLog(program));

    gl.disable(gl.DEPTH_TEST); // Disable depth testing

    // Enable blending
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

    // Attribute & uniform locations
    const a_pos   = gl.getAttribLocation(program, 'a_position');
    const a_col   = gl.getAttribLocation(program, 'a_color');
    const u_proj  = gl.getUniformLocation(program, 'projection');
    const u_view  = gl.getUniformLocation(program, 'view');
    // Create buffers (will be populated on worker init)
    const posBuff  = gl.createBuffer();
    const colBuff  = gl.createBuffer();
    const idxBuff = gl.createBuffer();
    let faceCount = 0, vertexCount = 0;

    // Spawn worker with our createWorker
    const worker = new Worker(URL.createObjectURL(
      new Blob(['(', createWorker.toString(), ')(self)'], { type: 'application/javascript' })
    ));

    // === Resize & Setup Projection ===
    let projectionMatrix;
    const resize = () => {
      canvas.width  = innerWidth;
      canvas.height = innerHeight;
      gl.viewport(0, 0, canvas.width, canvas.height);
      projectionMatrix = getProjectionMatrix(canvas.width/2, canvas.width/2, canvas.width, canvas.height);
      gl.uniformMatrix4fv(u_proj, false, projectionMatrix);
    };
    window.addEventListener('resize', resize);
    resize();


    // Handle worker messages
    let initialOrientationSet = false;
    function setInitialView(positionsBuffer) {
      const pos = new Float32Array(positionsBuffer);
      let minX=Infinity, minY=Infinity, minZ=Infinity;
      let maxX=-Infinity, maxY=-Infinity, maxZ=-Infinity;
      for (let i=0; i<pos.length; i+=3) {
        const x=pos[i], y=pos[i+1], z=pos[i+2];
        if(x<minX) minX=x; if(y<minY) minY=y; if(z<minZ) minZ=z;
        if(x>maxX) maxX=x; if(y>maxY) maxY=y; if(z>maxZ) maxZ=z;
      }
      const cx=(minX+maxX)/2, cy=(minY+maxY)/2, cz=(minZ+maxZ)/2;
      const rx=(maxX-minX), ry=(maxY-minY), rz=(maxZ-minZ);
      const r=Math.max(rx,ry,rz)/2;
      // place eye along +Z axis
      const eye=[cx + r * 0.3, cy, cz];
      // compute lookAt matrix
      const up=[0,1,0];
      const z0=eye[0]-cx, z1=eye[1]-cy, z2=eye[2]-cz;
      let len=Math.hypot(z0,z1,z2);
      const zv=[z0/len, z1/len, z2/len];
      const xv=[up[1]*zv[2]-up[2]*zv[1], up[2]*zv[0]-up[0]*zv[2], up[0]*zv[1]-up[1]*zv[0]];
      len=Math.hypot(...xv);
      xv[0]/=len; xv[1]/=len; xv[2]/=len;
      const yv=[zv[1]*xv[2]-zv[2]*xv[1], zv[2]*xv[0]-zv[0]*xv[2], zv[0]*xv[1]-zv[1]*xv[0]];
      viewMatrix = [
        xv[0], yv[0], zv[0], 0,
        xv[1], yv[1], zv[1], 0,
        xv[2], yv[2], zv[2], 0,
        -(xv[0]*eye[0]+xv[1]*eye[1]+xv[2]*eye[2]),
        -(yv[0]*eye[0]+yv[1]*eye[1]+yv[2]*eye[2]),
        -(zv[0]*eye[0]+zv[1]*eye[1]+zv[2]*eye[2]),
        1
      ];
      gl.uniformMatrix4fv(u_view, false, viewMatrix);
      initialOrientationSet = true;
    }

    let triPos = null, triCol = null;
    let elementIndices = null;

    // Handle worker messages
    worker.onmessage = e => {
      if (e.data.init) {
        // Received parsed PLY data: positions, faceIndices, faceColors, faceCount
        const positions = new Float32Array(e.data.positions);
        const faces     = new Uint32Array(e.data.faceIndices);
        const colors    = new Float32Array(e.data.faceColors);
        faceCount = e.data.faceCount;
        vertexCount = faceCount * 3;

        // Flatten positions and colors once
        triPos = new Float32Array(vertexCount * 3);
        triCol = new Float32Array(vertexCount * 4);
        for (let f = 0; f < faceCount; f++) {
            const baseI = f * 3;
            for (let v = 0; v < 3; v++) {
            const src = faces[baseI + v] * 3;
            const dstP = (baseI + v) * 3;
            triPos.set(positions.subarray(src, src + 3), dstP);
            const dstC = (baseI + v) * 4;
            triCol.set(colors.subarray(f * 4, f * 4 + 4), dstC);
            }
        }

        // Initial element array: identity
        elementIndices = new Uint32Array(vertexCount);
        for (let i = 0; i < faceCount; i++) {
            const b = i * 3;
            elementIndices[b] = b;
            elementIndices[b+1] = b+1;
            elementIndices[b+2] = b+2;
        }

        // Upload flattened positions
        gl.bindBuffer(gl.ARRAY_BUFFER, posBuff);
        gl.bufferData(gl.ARRAY_BUFFER, triPos, gl.STATIC_DRAW);
        gl.enableVertexAttribArray(a_pos);
        gl.vertexAttribPointer(a_pos, 3, gl.FLOAT, false, 0, 0);

        // Upload flattened colors
        gl.bindBuffer(gl.ARRAY_BUFFER, colBuff);
        gl.bufferData(gl.ARRAY_BUFFER, triCol, gl.STATIC_DRAW);
        gl.enableVertexAttribArray(a_col);
        gl.vertexAttribPointer(a_col, 4, gl.FLOAT, false, 0, 0);

        // Upload initial EBO
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, idxBuff);
        gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, elementIndices, gl.DYNAMIC_DRAW);

        setInitialView(e.data.positions);
        worker.postMessage({ viewProj: multiply4(projectionMatrix, viewMatrix) });
      }
      else if (e.data.depthOrder) {
        // Reorder elementIndices based on depthOrder
        const order = new Uint32Array(e.data.depthOrder);
        for (let i = 0; i < faceCount; i++) {
            const bDst = i * 3;
            const bSrc = order[i] * 3;
            elementIndices[bDst]   = bSrc;
            elementIndices[bDst+1] = bSrc+1;
            elementIndices[bDst+2] = bSrc+2;
        }
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, idxBuff);
        gl.bufferSubData(gl.ELEMENT_ARRAY_BUFFER, 0, elementIndices);
      }
    };

    let activeKeys = [];

    window.addEventListener("keydown", (e) => {
        // if (document.activeElement != document.body) return;
        if (!activeKeys.includes(e.code)) activeKeys.push(e.code);
    });
    window.addEventListener("keyup", (e) => {
        activeKeys = activeKeys.filter((k) => k !== e.code);
    });
    window.addEventListener("blur", () => {
        activeKeys = [];
    });

    window.addEventListener(
        "wheel",
        (e) => {
            e.preventDefault();
            const lineHeight = 10;
            const scale =
                e.deltaMode == 1
                    ? lineHeight
                    : e.deltaMode == 2
                      ? innerHeight
                      : 1;
            let inv = invert4(viewMatrix);
            if (e.shiftKey) {
                inv = translate4(
                    inv,
                    (e.deltaX * scale) / innerWidth,
                    (e.deltaY * scale) / innerHeight,
                    0,
                );
            } else if (e.ctrlKey || e.metaKey) {
                // inv = rotate4(inv,  (e.deltaX * scale) / innerWidth,  0, 0, 1);
                // inv = translate4(inv,  0, (e.deltaY * scale) / innerHeight, 0);
                // let preY = inv[13];
                inv = translate4(
                    inv,
                    0,
                    0,
                    (-10 * (e.deltaY * scale)) / innerHeight,
                );
                // inv[13] = preY;
            } else {
                let d = 4;
                inv = translate4(inv, 0, 0, d);
                inv = rotate4(inv, -(e.deltaX * scale) / innerWidth, 0, 1, 0);
                inv = rotate4(inv, (e.deltaY * scale) / innerHeight, 1, 0, 0);
                inv = translate4(inv, 0, 0, -d);
            }

            viewMatrix = invert4(inv);
        },
        { passive: false },
    );

    let startX, startY, down;
    canvas.addEventListener("mousedown", (e) => {
        e.preventDefault();
        startX = e.clientX;
        startY = e.clientY;
        down = e.ctrlKey || e.metaKey ? 2 : 1;
    });
    canvas.addEventListener("contextmenu", (e) => {
        e.preventDefault();
        startX = e.clientX;
        startY = e.clientY;
        down = 2;
    });

    canvas.addEventListener("mousemove", (e) => {
        e.preventDefault();
        if (down == 1) {
            let inv = invert4(viewMatrix);
            let dx = (5 * (e.clientX - startX)) / innerWidth;
            let dy = (5 * (e.clientY - startY)) / innerHeight;
            let d = 4;

            inv = translate4(inv, 0, 0, d);
            inv = rotate4(inv, dx, 0, 1, 0);
            inv = rotate4(inv, -dy, 1, 0, 0);
            inv = translate4(inv, 0, 0, -d);
            // let postAngle = Math.atan2(inv[0], inv[10])
            // inv = rotate4(inv, postAngle - preAngle, 0, 0, 1)
            // console.log(postAngle)
            viewMatrix = invert4(inv);

            startX = e.clientX;
            startY = e.clientY;
        } else if (down == 2) {
            let inv = invert4(viewMatrix);
            // inv = rotateY(inv, );
            // let preY = inv[13];
            inv = translate4(
                inv,
                (-10 * (e.clientX - startX)) / innerWidth,
                0,
                (10 * (e.clientY - startY)) / innerHeight,
            );
            // inv[13] = preY;
            viewMatrix = invert4(inv);

            startX = e.clientX;
            startY = e.clientY;
        }
    });
    canvas.addEventListener("mouseup", (e) => {
        e.preventDefault();
        down = false;
        startX = 0;
        startY = 0;
    });

    let altX = 0,
        altY = 0;
    canvas.addEventListener(
        "touchstart",
        (e) => {
            e.preventDefault();
            if (e.touches.length === 1) {
                startX = e.touches[0].clientX;
                startY = e.touches[0].clientY;
                down = 1;
            } else if (e.touches.length === 2) {
                // console.log('beep')
                startX = e.touches[0].clientX;
                altX = e.touches[1].clientX;
                startY = e.touches[0].clientY;
                altY = e.touches[1].clientY;
                down = 1;
            }
        },
        { passive: false },
    );
    canvas.addEventListener(
        "touchmove",
        (e) => {
            e.preventDefault();
            if (e.touches.length === 1 && down) {
                let inv = invert4(viewMatrix);
                let dx = (4 * (e.touches[0].clientX - startX)) / innerWidth;
                let dy = (4 * (e.touches[0].clientY - startY)) / innerHeight;

                let d = 4;
                inv = translate4(inv, 0, 0, d);
                // inv = translate4(inv,  -x, -y, -z);
                // inv = translate4(inv,  x, y, z);
                inv = rotate4(inv, dx, 0, 1, 0);
                inv = rotate4(inv, -dy, 1, 0, 0);
                inv = translate4(inv, 0, 0, -d);

                viewMatrix = invert4(inv);

                startX = e.touches[0].clientX;
                startY = e.touches[0].clientY;
            } else if (e.touches.length === 2) {
                // alert('beep')
                const dtheta =
                    Math.atan2(startY - altY, startX - altX) -
                    Math.atan2(
                        e.touches[0].clientY - e.touches[1].clientY,
                        e.touches[0].clientX - e.touches[1].clientX,
                    );
                const dscale =
                    Math.hypot(startX - altX, startY - altY) /
                    Math.hypot(
                        e.touches[0].clientX - e.touches[1].clientX,
                        e.touches[0].clientY - e.touches[1].clientY,
                    );
                const dx =
                    (e.touches[0].clientX +
                        e.touches[1].clientX -
                        (startX + altX)) /
                    2;
                const dy =
                    (e.touches[0].clientY +
                        e.touches[1].clientY -
                        (startY + altY)) /
                    2;
                let inv = invert4(viewMatrix);
                // inv = translate4(inv,  0, 0, d);
                inv = rotate4(inv, dtheta, 0, 0, 1);

                inv = translate4(inv, -dx / innerWidth, -dy / innerHeight, 0);

                // let preY = inv[13];
                inv = translate4(inv, 0, 0, 3 * (1 - dscale));
                // inv[13] = preY;

                viewMatrix = invert4(inv);

                startX = e.touches[0].clientX;
                altX = e.touches[1].clientX;
                startY = e.touches[0].clientY;
                altY = e.touches[1].clientY;
            }
        },
        { passive: false },
    );
    canvas.addEventListener(
        "touchend",
        (e) => {
            e.preventDefault();
            down = false;
            startX = 0;
            startY = 0;
        },
        { passive: false },
    );

    let lastFrame = 0;
    let avgFps = 0;
    let start = 0;

    window.addEventListener("gamepadconnected", (e) => {
        const gp = navigator.getGamepads()[e.gamepad.index];
        console.log(
            `Gamepad connected at index ${gp.index}: ${gp.id}. It has ${gp.buttons.length} buttons and ${gp.axes.length} axes.`,
        );
    });
    window.addEventListener("gamepaddisconnected", (e) => {
        console.log("Gamepad disconnected");
    });

    const frame = (now) => {
        if (vertexCount == 0) {
            gl.clear(gl.COLOR_BUFFER_BIT);
            requestAnimationFrame(frame);
            return;
        }

        let inv = invert4(viewMatrix);
        let shiftKey =
            activeKeys.includes("Shift") ||
            activeKeys.includes("ShiftLeft") ||
            activeKeys.includes("ShiftRight");

        if (activeKeys.includes("ArrowUp")) {
            if (shiftKey) {
                inv = translate4(inv, 0, -0.03, 0);
            } else {
                inv = translate4(inv, 0, 0, 0.1);
            }
        }
        if (activeKeys.includes("ArrowDown")) {
            if (shiftKey) {
                inv = translate4(inv, 0, 0.03, 0);
            } else {
                inv = translate4(inv, 0, 0, -0.1);
            }
        }
        if (activeKeys.includes("ArrowLeft"))
            inv = translate4(inv, -0.03, 0, 0);
        //
        if (activeKeys.includes("ArrowRight"))
            inv = translate4(inv, 0.03, 0, 0);
        // inv = rotate4(inv, 0.01, 0, 1, 0);
        if (activeKeys.includes("KeyA")) inv = rotate4(inv, -0.01, 0, 1, 0);
        if (activeKeys.includes("KeyD")) inv = rotate4(inv, 0.01, 0, 1, 0);
        if (activeKeys.includes("KeyQ")) inv = rotate4(inv, 0.01, 0, 0, 1);
        if (activeKeys.includes("KeyE")) inv = rotate4(inv, -0.01, 0, 0, 1);
        if (activeKeys.includes("KeyW")) inv = rotate4(inv, 0.005, 1, 0, 0);
        if (activeKeys.includes("KeyS")) inv = rotate4(inv, -0.005, 1, 0, 0);

        if (
            ["KeyJ", "KeyK", "KeyL", "KeyI"].some((k) => activeKeys.includes(k))
        ) {
            let d = 4;
            inv = translate4(inv, 0, 0, d);
            inv = rotate4(
                inv,
                activeKeys.includes("KeyJ")
                    ? -0.05
                    : activeKeys.includes("KeyL")
                      ? 0.05
                      : 0,
                0,
                1,
                0,
            );
            inv = rotate4(
                inv,
                activeKeys.includes("KeyI")
                    ? 0.05
                    : activeKeys.includes("KeyK")
                      ? -0.05
                      : 0,
                1,
                0,
                0,
            );
            inv = translate4(inv, 0, 0, -d);
        }

        viewMatrix = invert4(inv);

        const viewProj = multiply4(projectionMatrix, viewMatrix);
        worker.postMessage({ viewProj: viewProj });

        const currentFps = 1000 / (now - lastFrame) || 0;
        avgFps = avgFps * 0.9 + currentFps * 0.1;


        gl.clear(gl.COLOR_BUFFER_BIT);
        if (vertexCount > 0) {
            gl.uniformMatrix4fv(u_view, false, viewMatrix);
            gl.drawElements(gl.TRIANGLES, vertexCount, gl.UNSIGNED_INT, 0);
        }
        
        fps.innerText = Math.round(avgFps) + " fps";
        lastFrame = now;
        requestAnimationFrame(frame);
    };

    frame();

    // === isPly & selectFile ===
    const isPly = data => (
      data[0] === 0x70 && // 'p'
      data[1] === 0x6c && // 'l'
      data[2] === 0x79 && // 'y'
      data[3] === 0x0a    // '\n'
    );

    const selectFile = file => {
      const fr = new FileReader();
      fr.onload = () => {
        const buf = new Uint8Array(fr.result);
        if (isPly(buf)) {
          worker.postMessage({ ply: buf.buffer }, [buf.buffer]);
        } else {
          console.error('Not a PLY file');
        }
      };
      fr.readAsArrayBuffer(file);
    };

    const preventDefault = (e) => {
        e.preventDefault();
        e.stopPropagation();
    };
    document.addEventListener("dragenter", preventDefault);
    document.addEventListener("dragover", preventDefault);
    document.addEventListener("dragleave", preventDefault);
    document.addEventListener("drop", (e) => {
        e.preventDefault();
        e.stopPropagation();
        selectFile(e.dataTransfer.files[0]);
    });
}

main().catch((err) => {
    document.getElementById("message").innerText = err.toString();
});
