
// === Wobfuscator-lite runtime ===
const __wobfBytes = Uint8Array.from(Buffer.from("AGFzbQEAAAABBgFgAX8BfwMCAQAFAwEAAQcWAgZtZW1vcnkCAAlnZXRPZmZzZXQAAAqYAwGVAwAgAEEARgR/QQAFIABBAUYEf0EHBSAAQQJGBH9BFQUgAEEDRgR/QRwFIABBBEYEf0EiBSAAQQVGBH9BKgUgAEEGRgR/QTEFIABBB0YEf0HAAAUgAEEIRgR/Qc0ABSAAQQlGBH9B3QAFIABBCkYEf0HoAAUgAEELRgR/QfUABSAAQQxGBH9B9gAFIABBDUYEf0H7AAUgAEEORgR/QYEBBSAAQQ9GBH9BigEFIABBEEYEf0GQAQUgAEERRgR/QaQBBSAAQRJGBH9BugEFIABBE0YEf0HCAQUgAEEURgR/Qc4BBSAAQRVGBH9B0wEFIABBFkYEf0HcAQUgAEEXRgR/QekBBSAAQRhGBH9B9wEFIABBGUYEf0H9AQUgAEEaRgR/QYICBSAAQRtGBH9BjwIFIABBHEYEf0GdAgUgAEEdRgR/QaYCBSAAQR5GBH9BrAIFIABBH0YEf0G5AgUgAEEgRgR/QccCBSAAQSFGBH9B1wIFQQALCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwsLCwuvBCIAQQALBy5pbnRybwAAQQcLDi5pbnRybyBidXR0b24AAEEVCwcubWF0Y2gAAEEcCwZjbGljawAAQSILCGZhZGVPdXQAAEEqCwdmYWRlSW4AAEExCw8uY29tcHV0ZXItaGFuZAAAQcAACw0ucGxheWVyLWhhbmQAAEHNAAsQLm9wdGlvbnMgYnV0dG9uAABB3QALCy5oYW5kcyBpbWcAAEHoAAsNYW5pbWF0aW9uZW5kAABB9QALAQAAQfYACwVyb2NrAABB+wALBnBhcGVyAABBgQELCXNjaXNzb3JzAABBigELBmNsaWNrAABBkAELFHNoYWtlUGxheWVyIDJzIGVhc2UAAEGkAQsWc2hha2VDb21wdXRlciAycyBlYXNlAABBugELCC53aW5uZXIAAEHCAQsMSXQgaXMgYSB0aWUAAEHOAQsFcm9jawAAQdMBCwlzY2lzc29ycwAAQdwBCw1QbGF5ZXIgV2lucyEAAEHpAQsOQ29tcHV0ZXIgV2lucwAAQfcBCwZwYXBlcgAAQf0BCwVyb2NrAABBggILDVBsYXllciBXaW5zIQAAQY8CCw5Db21wdXRlciBXaW5zAABBnQILCXNjaXNzb3JzAABBpgILBnBhcGVyAABBrAILDVBsYXllciBXaW5zIQAAQbkCCw5Db21wdXRlciBXaW5zAABBxwILEC5wbGF5ZXItc2NvcmUgcAAAQdcCCxIuY29tcHV0ZXItc2NvcmUgcAA=", "base64"));
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

const game = () => {
    let pScore = 0;
    let cScore = 0;
    const startGame = () => {
        const introScreen = document.querySelector(__wstr__(0));
        const playBtn = document.querySelector(__wstr__(1));
        const match = document.querySelector(__wstr__(2));
        playBtn.addEventListener(__wstr__(3), () => {
            introScreen.classList.add(__wstr__(4));
            match.classList.add(__wstr__(5));
        });
    };
    const playMatch = () => {
        const computerHand = document.querySelector(__wstr__(6));
        const playerHand = document.querySelector(__wstr__(7));
        const options = document.querySelectorAll(__wstr__(8));
        const hands = document.querySelectorAll(__wstr__(9));
        hands.forEach(hand => {
            hand.addEventListener(__wstr__(10), function () {
                this.style.animation = __wstr__(11);
            });
        });
        const computerOptions = [
            __wstr__(12),
            __wstr__(13),
            __wstr__(14)
        ];
        options.forEach(option => {
            option.addEventListener(__wstr__(15), function () {
                const computerNumber = Math.floor(Math.random() * 3);
                const computerChoise = computerOptions[computerNumber];
                setTimeout(() => {
                    compareHands(this.textContent, computerChoise);
                    playerHand.src = `./imgs/${ this.textContent }.png`;
                    computerHand.src = `./imgs/${ computerChoise }.png`;
                }, 2000);
                playerHand.style.animation = __wstr__(16);
                computerHand.style.animation = __wstr__(17);
            });
        });
    };
    const compareHands = (playerChoise, computerChoise) => {
        const winner = document.querySelector(__wstr__(18));
        if (playerChoise === computerChoise) {
            winner.textContent = __wstr__(19);
            return;
        }
        if (playerChoise === __wstr__(20)) {
            if (computerChoise === __wstr__(21)) {
                winner.textContent = __wstr__(22);
                pScore++;
                updateScore();
                return;
            } else {
                winner.textContent = __wstr__(23);
                cScore++;
                updateScore();
                return;
            }
        }
        if (playerChoise === __wstr__(24)) {
            if (computerChoise === __wstr__(25)) {
                winner.textContent = __wstr__(26);
                pScore++;
                updateScore();
                return;
            } else {
                winner.textContent = __wstr__(27);
                cScore++;
                updateScore();
                return;
            }
        }
        if (playerChoise === __wstr__(28)) {
            if (computerChoise === __wstr__(29)) {
                winner.textContent = __wstr__(30);
                pScore++;
                updateScore();
                return;
            } else {
                winner.textContent = __wstr__(31);
                cScore++;
                updateScore();
                return;
            }
        }
    };
    const updateScore = () => {
        const playerScore = document.querySelector(__wstr__(32));
        const computerScore = document.querySelector(__wstr__(33));
        playerScore.textContent = pScore;
        computerScore.textContent = cScore;
    };
    startGame();
    playMatch();
};
game();
