// wobfuscator.js
//
// Wobfuscator-lite: move all string literals into a WASM module
// and replace them with __wstr(index) calls.

const esprima = require('esprima');
const estraverse = require('estraverse');
const escodegen = require('escodegen');
const wabtFactory = require('wabt');

/**
 * Build a tiny WASM module from a list of strings.
 * Each string is stored in linear memory, null-terminated.
 * Exported function: getOffset(i) -> byte offset of string i.
 */
async function generateWasmBase64(strings) {
    const wabt = await wabtFactory();

    let offset = 0;
    const offsets = [];
    let dataSegments = '';

    for (const s of strings) {
        const bytes = Buffer.from(s + '\0', 'utf8');
        const escaped = Array.from(bytes)
            .map(b => '\\' + b.toString(16).padStart(2, '0'))
            .join('');

        dataSegments += `  (data (i32.const ${offset}) "${escaped}")\n`;
        offsets.push(offset);
        offset += bytes.length;
    }

    // Build nested if/else expression for getOffset
    function buildOffsetIfChain(i) {
        if (i >= offsets.length) {
            return '    (i32.const 0)'; // default
        }
        return `
    (if (result i32) (i32.eq (local.get $i) (i32.const ${i}))
      (then (i32.const ${offsets[i]}))
      (else
${buildOffsetIfChain(i + 1)}
      )
    )`;
    }

    const wat = `
  (module
    (memory (export "memory") 1)
    (func (export "getOffset") (param $i i32) (result i32)
${offsets.length > 0 ? buildOffsetIfChain(0) : '      (i32.const 0)'}
    )
${dataSegments}
  )
  `;

    const wasmModule = wabt.parseWat('wobf.wat', wat);
    const { buffer } = wasmModule.toBinary({});
    return Buffer.from(buffer).toString('base64');
}

/**
 * Transform JS source:
 * - Collect string literals
 * - Replace them with __wstr(index)
 * - Generate WASM module that stores all strings
 * - Prepend runtime glue to use the WASM
 */
async function wobfuscateSource(sourceCode) {
    const ast = esprima.parseScript(sourceCode, { range: true });

    const strings = [];

    // Replace all string literals with __wstr(index)
    const newAst = estraverse.replace(ast, {
        enter(node) {
            if (node.type === 'Literal' && typeof node.value === 'string') {
                const idx = strings.length;
                strings.push(node.value);
                return {
                    type: 'CallExpression',
                    callee: { type: 'Identifier', name: '__wstr__' },
                    arguments: [{ type: 'Literal', value: idx }]
                };
            }
        }
    });

    const wasmBase64 = await generateWasmBase64(strings);

    const runtimeHeader = `
// === Wobfuscator-lite runtime ===
const __wobfBytes = Uint8Array.from(Buffer.from("${wasmBase64}", "base64"));
const __wobfModule = new WebAssembly.Module(__wobfBytes);
const __wobfInstance = new WebAssembly.Instance(__wobfModule);
const __wobfMemory = __wobfInstance.exports.memory;
const __wobfGetOffset = __wobfInstance.exports.getOffset;

function __wstr__(idx) {
    const mem = new Uint8Array(__wobfMemory.buffer);
    let off = __wobfGetOffset(idx);
    const chars = [];
    while (mem[off] !== 0) {
        chars.push(mem[off++]);
    }
    return String.fromCharCode(...chars);
}
// === End Wobfuscator-lite runtime ===

`;

    const transformedJs = escodegen.generate(newAst);
    return runtimeHeader + transformedJs + '\n';
}

module.exports = {
    wobfuscateSource
};
