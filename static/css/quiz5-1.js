// questionnaire for depression
const questionnaire=[
    {

        question:"1. Stomach Pain",
        a:"0",
        b:"1",
        c:"2",
    },
    {
        question:"2. Back Pain",
        a:"0",
        b:"1",
        c:"2",
       
    },
    {

        question:"3. Pain in your arms, legs, or joints(knees,hips,etc)",
        a:"0",
        b:"1",
        c:"2",
       
    },
    {

        question:"4. Menstrual Cramps or any other problems with your periods (Women Only)",
        a:"0",
        b:"1",
        c:"2",
       
    },
    {

        question:"5. Headaches",
        a:"0",
        b:"1",
        c:"2",
        
    },
    {

        question:"6. Chest Pain",
        a:"0",
        b:"1",
        c:"2",
       
    },
    {

        question:"7. Dizziness",
        a:"0",
        b:"1",
        c:"2",
       
    },
    {

        question:"8. Fainting sleeps",
        a:"0",
        b:"1",
        c:"2"
       
    },
    {

        question:"9. Feeling your heart pound or race",
        a:"0",
        b:"1",
        c:"2"
    },
    {

        question:"10. Shortness of Breath",
        a:"0",
        b:"1",
        c:"2"
    },
    {

        question:"11. Pain or problems during sexual intercourse",
        a:"0",
        b:"1",
        c:"2"
    },
    {

        question:"12. Constipation; loose bowels, or diarrhea",
        a:"0",
        b:"1",
        c:"2"
    },
    {

        question:"13. Nausea, gas or indigestion",
        a:"0",
        b:"1",
        c:"2"
    },
    {

        question:"14. Feeling tired or having low energy",
        a:"0",
        b:"1",
        c:"2"
    },
    {

        question:"15. Trouble Sleeping",
        a:"0",
        b:"1",
        c:"2"
    }
];

const question= document.querySelector('.question');
const option1=document.querySelector('#option1')
const option2=document.querySelector('#option2')
const option3=document.querySelector('#option3')
const next=document.querySelector('#next')
const answers=document.querySelectorAll('.answer')
const showScore=document.querySelector('#showScore')
const showProblem=document.querySelector('#showProblem')
const mainq=document.querySelector('#mainq')
const optionId=document.querySelector('#optionId')
const submit=document.querySelector('#submit')
const greet=document.querySelector('#greet')
const note=document.querySelector('#note')

let quesCount=0;
var score=0;
const loadQuestion=()=>{
    const questionList= questionnaire[quesCount]
    question.innerHTML=questionList.question;
    option1.innerHTML=questionList.a;
    option2.innerHTML=questionList.b;
    option3.innerHTML=questionList.c;
    // option4.innerHTML=questionList.d;
}

loadQuestion();

const getCheckAnswer=()=>{
    let answer;
    answers.forEach((currAnsElem)=>{
        if(currAnsElem.checked){
            answer=currAnsElem.id;
        }
    });
    return answer;
};

const deSelectAll=()=>{
    answers.forEach((currAnsElem)=>currAnsElem.checked=false)
}
submit.disabled=true

next.addEventListener('click',()=>{
    // greet.innerHTML=``;
    // note.innerHTML=``;
    const checkedAnswer=getCheckAnswer();
    const result=console.log(checkedAnswer);

    if(checkedAnswer==0){
        score+=0
    }
    else if(checkedAnswer==1){
        score+=1
    }
    else if(checkedAnswer==2){
        score+=2
    }
    // else if(checkedAnswer==3){
    //     score+=3
    // }
    // else{
    //     score=score
    // }
    
    else{
        a=confirm('Do you want to skip this question?')
        if(a){
            loadQuestion();
        }
        else{
            quesCount-=1;
            loadQuestion();
            deSelectAll();
        }
    }
    quesCount++;
    deSelectAll();
    
    // else{
    //     a=confirm('Do you want to skip this question')
    //     if(a){
    //         alert('Question Skipped')
    //     }
    //     else{
    //         quesCount-=1;
    //         loadQuestion();
    //         deSelectAll();
    //     }
    // }
    
    if(quesCount<questionnaire.length-1){
        loadQuestion();
    }
    else if(quesCount<questionnaire.length){
        next.innerHTML=`<h3>Submit</h3>`
        loadQuestion();
        submit.disabled=false;
        next.disabled=true;
        next.innerHTML=`<h3>End of Questions</h3>`
    }
}),

submit.addEventListener('click',()=>{
    // submit.addEventListener("hover",disable1);
    let confirmation=confirm('Do you want to submit?')
    if(confirmation){
    const checkedAnswer=getCheckAnswer();
    const result=console.log(checkedAnswer);

   if(checkedAnswer==0){
        score+=0
    }
    else if(checkedAnswer==1){
        score+=1
    }
    else if(checkedAnswer==2){
        score+=2
    }
    // else if(checkedAnswer==3){
    //     score+=3
    // }
    else{
        score=score
    }
        //for blanking
        optionId.innerHTML=``
        mainq.innerHTML=``  
        quesId.innerHTML=``
        submit.disabled=true
        submit.innerHTML=`<h3>Test Submitted</h3>`
        next.disabled=true
        next.innerHTML=`<h3>
        Wohoo!, you did it! Congrats on completing the quiz and taking a big step towards improving your mental health. By clicking the "Solution" button, you'll be taken to a treasure trove of solutions tailored to your specific needs. Get ready to embark on a journey towards better mental wellness - good luck! 🚀
        
        </h3>
        `
        
        const val=score
        console.log(val)
        if(val>=0 && val<=4){
            submit.innerHTML=`<h3>Severity level: None to slight </h3>`
        }
        else if(val>5 && val<=9){
            submit.innerHTML=`<h3>Severity level: Mild</h3>`
        }
        else if(val>10 && val<=14){
            submit.innerHTML=`<h3>Severity level: Moderate</h3>`
        }
        else{
            submit.innerHTML=`<h3>Severity level: Severe</h3>`
        }
        // else{
        //     submit.innerHTML=`<h4>Severe Depression</h4>`;
        // }
        showProblem.innerHTML=`<button class="btn" onclick="location.reload()">Give Test Again </button> <br> <button class="btn" onclick="window.location.href='/solution5'">See Solutions</button> <br>`
        showProblem.classList.remove('problemArea')
    }
    else{
        alert('Quiz was not submitted')
    }

    }

);